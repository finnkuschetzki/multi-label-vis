import { data } from "@/stores/data.js"
import {
    dataPointGroundTruth,
    dataPointImagePath,
    dataPointPredictions,
    overlayPosition,
    showOverlay
} from "@/stores/overlay.js"

import { margin, tableau20 } from "@/chart/settings.js"
import { getGlyphSize, getNumClasses, getStrokeWidth } from "@/chart/units.js";

import * as settings from "@/stores/settings.js"



/* CALCULATIONS */


function calculateGlyphData(xScale, yScale, glyphSize, glyphBoundingSize, featureColumn) {
    const numClasses = getNumClasses()

    const glyphData = []

    const initialRadians = 3 / 2 * Math.PI
    const classStep = (2 * Math.PI) / numClasses

    data.value.forEach(d => {
        // setting focused if there is at least one focus index
        let focusFeature, focused

        if (settings.glyphData.value === "groundTruth") {
            focusFeature = "ground_truth"
        } else if (settings.glyphData.value === "predictions") {
            focusFeature = "binarized_predictions"
        }

        if (!focusFeature || settings.focusIndices.value.length === 0) {
            focused = true
        } else if (settings.focusSetOperation.value === "union") {
            focused = settings.focusIndices.value.some(i => d[focusFeature][i])
        } else if (settings.focusSetOperation.value === "intersection") {
            focused = settings.focusIndices.value.every(i => d[focusFeature][i])
        } else {
            throw Error("incorrect value for focusSetOperation")
        }

        // calculating glyph data
        const x = xScale(d[featureColumn][0]) + margin.left
        const y = yScale(d[featureColumn][1]) + margin.bottom

        const cx = x + glyphBoundingSize/2
        const cy = y + glyphBoundingSize/2

        const circlePoints = []
        for (let i = 0; i < numClasses; i++) {
            circlePoints.push([
                cx + Math.cos(initialRadians + classStep * i) * glyphSize/2,  // x pos
                cy + Math.sin(initialRadians + classStep * i) * glyphSize/2  // y pos
            ])
        }

        const segments = []
        for (let i = 0; i < circlePoints.length; i++) {
            segments.push({
                "classIndex": i,
                "centerPoint": [cx, cy],
                "outerPoints": [circlePoints[i], circlePoints[(i+1) % numClasses]],
                "groundTruth": d["ground_truth"][i],
                "prediction": d["predictions"][i],
                "binarizedPrediction": d["binarized_predictions"][i],
                "focused": focused
            })
        }

        glyphData.push({
            "x": x,
            "y": y,
            "cx": cx,
            "cy": cy,
            "circlePoints": circlePoints,
            "segments": segments,
            "imagePath": d["image_path"],
            "groundTruth": d["ground_truth"],
            "predictions": d["predictions"],
            "focused": focused
        })
    })

    const segmentData = glyphData.flatMap(glyph => glyph.segments)

    return { glyphData, segmentData }
}



/* PARTS OF GLYPHS */


function clearGlyphs(contentGroup) {
    // todo change name of glyph-lines (does not fit for simple glyph)
    contentGroup.selectAll(".glyph-lines").remove()
    contentGroup.selectAll(".glyph-segment-lines").remove()
    contentGroup.selectAll(".glyph-segment-fills").remove()
    contentGroup.selectAll(".glyph-whiskers").remove()
    contentGroup.selectAll(".glyph-event-box").remove()
}


function drawCircles(contentGroup, glyphSize, glyphData) {
    contentGroup.selectAll(".glyph-lines")
        .data(glyphData)
        .enter()
        .append("circle")
        .attr("class", "glyph-lines")
        .attr("cx", g => g.cx)
        .attr("cy", g => g.cy)
        .attr("r", glyphSize / 2)
        .attr("stroke", "none")
        .attr("fill", g => g.focused ? "black" : "lightgray")
}


function drawGlyphLines(contentGroup, glyphData, strokeWidth) {
    contentGroup.selectAll(".glyph-lines")
        .data(glyphData)
        .enter()
        .append("path")
        .attr("class", "glyph-lines")
        .attr("d", g => {
            let d = ``

            // segment borders
            for (let i = 0; i < g.circlePoints.length; i++) {
                d += `M${g.cx},${g.cy}`
                d += `L${g.circlePoints[i][0]},${g.circlePoints[i][1]}`
            }

            // outline
            d += `M${g.circlePoints[0][0]},${g.circlePoints[0][1]}`  // move to first point
            for (let i = 1; i < g.circlePoints.length; i++) {
                d += `L${g.circlePoints[i][0]},${g.circlePoints[i][1]}`  // lines to other points
            }
            d += `Z`  // complete to first point

            return d
        })
        .attr("stroke", g => g.focused ? "black" : "silver")
        .attr("stroke-width", strokeWidth)
        .attr("fill", "none")
}


function groundTruthColor(s) {
    return s.groundTruth ? tableau20[s.classIndex] : "none"
}


function binarizedPredictionColor(s) {
    return s.binarizedPrediction ? tableau20[s.classIndex] : "none"
}


function comparisonColor(s) {
    if (s.groundTruth && s.binarizedPrediction) {
        return "darkgray"
    } else if (s.groundTruth) {
        return "red"
    } else if (s.binarizedPrediction) {
        return "dodgerblue"
    } else {
        return "none"
    }
}


function standardOpacity(s) {
    return 1.0
}


function focusOpacity(s) {
    return s.focused ? 1.0 : 0.3
}


function comparisonOpacity(s) {
    if (s.binarizedPrediction) {
        return s.prediction / 2
    } else {
        return 1 - s.prediction * 2
    }
}


function drawGlyphFillsBinary(contentGroup, segmentData, fillColorFunc, fillOpacityFunc) {
    contentGroup.selectAll(".glyph-segment-fills")
        .data(segmentData)
        .enter()
        .append("path")
        .attr("class", "glyph-segment-fills")
        .attr("d", s => {
            let d = ``

            // segment fill
            d += `M${s.centerPoint[0]},${s.centerPoint[1]}`
            d += `L${s.outerPoints[0][0]},${s.outerPoints[0][1]}`
            d += `L${s.outerPoints[1][0]},${s.outerPoints[1][1]}`
            d += `Z`

            return d
        })
        .attr("stroke", "none")
        .attr("fill", s => fillColorFunc(s))
        .attr("fill-opacity", s => fillOpacityFunc(s))
}


function drawGlyphFillsPartial(contentGroup, segmentData) {
    contentGroup.selectAll(".glyph-segment-fills")
        .data(segmentData)
        .enter()
        .append("path")
        .attr("class", "glyph-segment-fills")
        .attr("d", s => {
            // vectors from center point to outer points
            const vec0 = [s.outerPoints[0][0] - s.centerPoint[0], s.outerPoints[0][1] - s.centerPoint[1]]
            const vec1 = [s.outerPoints[1][0] - s.centerPoint[0], s.outerPoints[1][1] - s.centerPoint[1]]

            // partial fill points
            const fillPoint0 = [s.centerPoint[0] + s.prediction * vec0[0], s.centerPoint[1] + s.prediction * vec0[1]]
            const fillPoint1 = [s.centerPoint[0] + s.prediction * vec1[0], s.centerPoint[1] + s.prediction * vec1[1]]

            let d = ``

            // segment fill
            d += `M${s.centerPoint[0]},${s.centerPoint[1]}`
            d += `L${fillPoint0[0]},${fillPoint0[1]}`
            d += `L${fillPoint1[0]},${fillPoint1[1]}`
            d += `Z`

            return d
        })
        .attr("stroke", "none")
        // todo cutoff at 0.1 prediction to increase performance of web app
        .attr("fill", s => s.prediction >= 0.1 ? tableau20[s.classIndex] : "none")
        .attr("fill-opacity", s => s.focused ? 1.0 : 0.3)
}


function drawGlyphFillsSegments(contentGroup, segmentData) {
    // segment fill
    contentGroup.selectAll(".glyph-segment-fills")
        .data(segmentData)
        .enter()
        .append("path")
        .attr("class", "glyph-segment-fills")
        .attr("d", s => {
            // calculate approximate fill
            const closestQuintile = Math.round(s.prediction * 5) / 5

            // draw approximate fill
            let d = ``

            if (closestQuintile > 0) {
                // vectors from center point to outer points
                const vec0 = [s.outerPoints[0][0] - s.centerPoint[0], s.outerPoints[0][1] - s.centerPoint[1]]
                const vec1 = [s.outerPoints[1][0] - s.centerPoint[0], s.outerPoints[1][1] - s.centerPoint[1]]

                // segment fill points
                const fillPoint0 = [s.centerPoint[0] + closestQuintile * vec0[0], s.centerPoint[1] + closestQuintile * vec0[1]]
                const fillPoint1 = [s.centerPoint[0] + closestQuintile * vec1[0], s.centerPoint[1] + closestQuintile * vec1[1]]

                d += `M${s.centerPoint[0]},${s.centerPoint[1]}`
                d += `L${fillPoint0[0]},${fillPoint0[1]}`
                d += `L${fillPoint1[0]},${fillPoint1[1]}`
                d += `Z`
            }

            return d
        })
        .attr("stroke", "none")
        .attr("fill", s => tableau20[s.classIndex])
        .attr("fill-opacity", s => s.focused ? 1.0 : 0.3)
}


function drawGlyphSegmentLines(contentGroup, glyphData, strokeWidth) {
    contentGroup.selectAll(".glyph-segment-lines")
        .data(glyphData)
        .enter()
        .append("path")
        .attr("class", "glyph-segment-lines")
        .attr("d", g => {
            // vectors from center point to circle points
            const circlePointsVec = g.circlePoints.map(cp => [cp[0] - g.cx, cp[1] - g.cy])

            let d = ``

            // draw three circles of segment lines
            for (let i = 1; i <= 4; i++) {

                d += `M${g.cx + i/5 * circlePointsVec[0][0]},${g.cy + i/5 * circlePointsVec[0][1]}`  // move to first point
                for (let j = 1; j < circlePointsVec.length; j++) {
                    d += `L${g.cx + i/5 * circlePointsVec[j][0]},${g.cy + i/5 * circlePointsVec[j][1]}`  // lines to other points
                }
                d += `Z`  // complete to first point

            }

            return d
        })
        .attr("stroke", "silver")
        .attr("stroke-width", strokeWidth)
        .attr("fill", "none")
}


function drawGlyphWhiskers(contentGroup, segmentData, strokeWidth) {
    contentGroup.selectAll(".glyph-whiskers")
        .data(segmentData)
        .enter()
        .append("path")
        .attr("class", "glyph-whiskers")
        .attr("d", s => {
            // todo cutoff at 0.1 prediction to increase performance of web app
            if (s.prediction >= 0.1) {
                // relative vector from outerPoint 0 to outerPoint 1
                const relOuterVec = [
                    s.outerPoints[1][0] - s.outerPoints[0][0],
                    s.outerPoints[1][1] - s.outerPoints[0][1]
                ]

                // relative vector from center point to outerPoint 0
                const relOuterPoint0Vec = [
                    s.outerPoints[0][0] - s.centerPoint[0],
                    s.outerPoints[0][1] - s.centerPoint[1]
                ]

                // relative vector from center point to midpoint between outerPoints
                const relMidpointVec = [
                    relOuterPoint0Vec[0] + relOuterVec[0]/2,
                    relOuterPoint0Vec[1] + relOuterVec[1]/2
                ]

                // absolute vector to whisker (outer) endpoint
                const whiskerEndpointVec = [
                    s.centerPoint[0] + s.prediction * relMidpointVec[0],
                    s.centerPoint[1] + s.prediction * relMidpointVec[1]
                ]

                let d = ``

                // whisker
                d += `M${s.centerPoint[0]},${s.centerPoint[1]}`
                d += `L${whiskerEndpointVec[0]},${whiskerEndpointVec[1]}`

                return d
            } else {
                return ""
            }
        })
        .attr("stroke", s => s.focused ? "black" : "silver")
        .attr("stroke-width", strokeWidth)
        .attr("fill", "none")
}



/* EVENTS */


export function overlayOnClick(contentGroup, glyphSize, glyphData) {
    contentGroup.selectAll(".glyph-event-box")
        .data(glyphData)
        .enter()
        .append("rect")
        .attr("class", "glyph-event.box")
        .attr("x", d => d.x)
        .attr("y", d => d.y)
        .attr("width", glyphSize)
        .attr("height", glyphSize)
        .attr("stroke", "none")
        .attr("fill", "none")
        .attr("pointer-events", "all")
        .on("click", (event, d) => {
            showOverlay.value = true
            dataPointImagePath.value = d.imagePath
            dataPointGroundTruth.value = d.groundTruth
            dataPointPredictions.value = d.predictions
            // calculate overlayPosition
            const chartBoundingClientRect = document.getElementById("chart").getBoundingClientRect()
            const chartMidHeight = (chartBoundingClientRect.y + chartBoundingClientRect.height) / 2
            if (event.clientY < chartMidHeight) {
                overlayPosition.value = "bottom"
            } else {
                overlayPosition.value = "top"
            }
        })
}



/* FULL GLYPHS */


export function drawSimpleGlyphs(contentGroup, xScale, yScale, feature_column) {
    const { glyphSize, glyphBoundingSize } = getGlyphSize(xScale, yScale)
    const { glyphData } = calculateGlyphData(xScale, yScale, glyphSize, glyphBoundingSize, feature_column)

    // removing old glyphs
    clearGlyphs(contentGroup)

    // simple glyphs
    drawCircles(contentGroup, glyphSize, glyphData)

    // details overlay on click
    overlayOnClick(contentGroup, glyphSize, glyphData)
}


export function drawGroundTruthGlyphs(contentGroup, xScale, yScale, feature_column) {
    const { glyphSize, glyphBoundingSize } = getGlyphSize(xScale, yScale)
    const { glyphData, segmentData } = calculateGlyphData(xScale, yScale, glyphSize, glyphBoundingSize, feature_column)
    const strokeWidth = getStrokeWidth()

    // removing old glyphs
    clearGlyphs(contentGroup)

    // segment fills
    drawGlyphFillsBinary(contentGroup, segmentData, groundTruthColor, focusOpacity)

    // glyph lines (outline and segment borders)
    drawGlyphLines(contentGroup, glyphData, strokeWidth)

    // details overlay on click
    overlayOnClick(contentGroup, glyphSize, glyphData)
}


export function drawBinaryGlyphs(contentGroup, xScale, yScale, feature_column) {
    const { glyphSize, glyphBoundingSize } = getGlyphSize(xScale, yScale)
    const { glyphData, segmentData } = calculateGlyphData(xScale, yScale, glyphSize, glyphBoundingSize, feature_column)
    const strokeWidth = getStrokeWidth()

    // removing old glyphs
    clearGlyphs(contentGroup)

    // segment fills
    drawGlyphFillsBinary(contentGroup, segmentData, binarizedPredictionColor, focusOpacity)

    // glyph lines (outline and segment borders)
    drawGlyphLines(contentGroup, glyphData, strokeWidth)

    // details overlay on click
    overlayOnClick(contentGroup, glyphSize, glyphData)
}


export function drawPartialFillGlyphs(contentGroup, xScale, yScale, feature_column) {
    const { glyphSize, glyphBoundingSize } = getGlyphSize(xScale, yScale)
    const { glyphData, segmentData } = calculateGlyphData(xScale, yScale, glyphSize, glyphBoundingSize, feature_column)
    const strokeWidth = getStrokeWidth()

    // removing old glyphs
    clearGlyphs(contentGroup)

    // segment fills
    drawGlyphFillsPartial(contentGroup, segmentData)

    // glyph lines (outline and segment borders)
    drawGlyphLines(contentGroup, glyphData, strokeWidth)

    // details overlay on click
    overlayOnClick(contentGroup, glyphSize, glyphData)
}


export function drawSegmentFillGlyphs(contentGroup, xScale, yScale, feature_column) {
    const { glyphSize, glyphBoundingSize } = getGlyphSize(xScale, yScale)
    const { glyphData, segmentData } = calculateGlyphData(xScale, yScale, glyphSize, glyphBoundingSize, feature_column)
    const strokeWidth = getStrokeWidth()

    // removing old glyphs
    clearGlyphs(contentGroup)

    // segment fills
    drawGlyphFillsSegments(contentGroup, segmentData)

    // segment lines
    drawGlyphSegmentLines(contentGroup, glyphData, strokeWidth)

    // glyph lines (outline and segment borders)
    drawGlyphLines(contentGroup, glyphData, strokeWidth)

    // details overlay on click
    overlayOnClick(contentGroup, glyphSize, glyphData)
}


export function drawWhiskerGlyphs(contentGroup, xScale, yScale, feature_column) {
    const { glyphSize, glyphBoundingSize } = getGlyphSize(xScale, yScale)
    const { glyphData, segmentData } = calculateGlyphData(xScale, yScale, glyphSize, glyphBoundingSize, feature_column)
    const strokeWidth = getStrokeWidth()

    // removing old glyphs
    clearGlyphs(contentGroup)

    // segment fills
    drawGlyphFillsBinary(contentGroup, segmentData, binarizedPredictionColor, focusOpacity)

    // glyph lines (outline and segment borders)
    drawGlyphLines(contentGroup, glyphData, strokeWidth)

    // whiskers
    drawGlyphWhiskers(contentGroup, segmentData, strokeWidth)

    // details overlay on click
    overlayOnClick(contentGroup, glyphSize, glyphData)
}


export function drawBinaryComparisonGlyphs(contentGroup, xScale, yScale, feature_column) {
    const { glyphSize, glyphBoundingSize } = getGlyphSize(xScale, yScale)
    const { glyphData, segmentData } = calculateGlyphData(xScale, yScale, glyphSize, glyphBoundingSize, feature_column)
    const strokeWidth = getStrokeWidth()

    // removing old glyphs
    clearGlyphs(contentGroup)

    // segment fills
    drawGlyphFillsBinary(contentGroup, segmentData, comparisonColor, standardOpacity)

    // glyph lines (outline and segment borders)
    drawGlyphLines(contentGroup, glyphData, strokeWidth)

    // details overlay on click
    overlayOnClick(contentGroup, glyphSize, glyphData)
}


export function drawOpacityComparisonGlyphs(contentGroup, xScale, yScale, feature_column) {
    const { glyphSize, glyphBoundingSize } = getGlyphSize(xScale, yScale)
    const { glyphData, segmentData } = calculateGlyphData(xScale, yScale, glyphSize, glyphBoundingSize, feature_column)
    const strokeWidth = getStrokeWidth()

    // removing old glyphs
    clearGlyphs(contentGroup)

    // segment fills
    drawGlyphFillsBinary(contentGroup, segmentData, comparisonColor, comparisonOpacity)

    // glyph lines (outline and segment borders)
    drawGlyphLines(contentGroup, glyphData, strokeWidth)

    // details overlay on click
    overlayOnClick(contentGroup, glyphSize, glyphData)
}
