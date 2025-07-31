import config from "../../../config.json"

import { data } from "@/stores/data.js"
import * as settings from "@/stores/settings.js"
import { showOverlay, overlayPosition, dataPointGroundTruth, dataPointPredictions, dataPointImagePath } from "@/stores/overlay.js"
import { margin, glyphSizeMultiplier, tableau20 } from "@/chart/settings.js"



/* CALCULATIONS */


function getNumClasses() {
    return data.value[0]["ground_truth"].length
}


function getGlyphSize(xScale, yScale) {
    const desiredGlyphSize = config.models.find(m => m.name === settings.modelName.value)["glyphSize"][settings.dataType.value]
    const glyphBoundingSize = Math.min(
        (xScale(desiredGlyphSize) - xScale(0)),
        (yScale(desiredGlyphSize) - yScale(0))
    )
    const glyphSize = glyphBoundingSize * glyphSizeMultiplier
    return { glyphSize, glyphBoundingSize }
}


function getStrokeWidth() {
    const desiredGlyphSize = config.models.find(m => m.name === settings.modelName.value)["glyphSize"][settings.dataType.value]
    return desiredGlyphSize * 10
}


function calculateGlyphData(xScale, yScale, glyphSize, glyphBoundingSize, featureColumn) {
    const numClasses = getNumClasses()

    const glyphData = []

    const initialRadians = 3 / 2 * Math.PI
    const classStep = (2 * Math.PI) / numClasses

    data.value.forEach(d => {
        const cx = xScale(d[featureColumn][0]) + margin.left
        const cy = yScale(d[featureColumn][1]) + margin.bottom

        const mx = cx + glyphBoundingSize/2
        const my = cy + glyphBoundingSize/2

        const circlePoints = []
        for (let i = 0; i < numClasses; i++) {
            circlePoints.push([
                mx + Math.cos(initialRadians + classStep * i) * glyphSize/2,  // x pos
                my + Math.sin(initialRadians + classStep * i) * glyphSize/2  // y pos
            ])
        }

        const segments = []
        for (let i = 0; i < circlePoints.length; i++) {
            segments.push({
                "classIndex": i,
                "centerPoint": [mx, my],
                "outerPoints": [circlePoints[i], circlePoints[(i+1) % numClasses]],
                "groundTruth": d["ground_truth"][i],
                "prediction": d["predictions"][i],
                "binarizedPrediction": d["binarized_predictions"][i]
            })
        }

        glyphData.push({
            "cx": cx,
            "cy": cy,
            "mx": mx,
            "my": my,
            "circlePoints": circlePoints,
            "segments": segments,
            "imagePath": d["image_path"],
            "groundTruth": d["ground_truth"],
            "predictions": d["predictions"]
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
        .attr("fill", "black")
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
                d += `M${g.mx},${g.my}`
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
        .attr("stroke", "black")
        .attr("stroke-width", strokeWidth)
        .attr("fill", "none")
}


function drawGlyphFillsBinary(contentGroup, segmentData) {
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
        .attr("fill", s => s.binarizedPrediction ? tableau20[s.classIndex] : "none")
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
}


function drawGlyphFillsSegments(contentGroup, segmentData, strokeWidth) {
    // segment fill
    contentGroup.selectAll(".glyph-segment-fills")
        .data(segmentData)
        .enter()
        .append("path")
        .attr("class", "glyph-segment-fills")
        .attr("d", s => {
            // calculate approximate fill
            const closestQuarter = Math.round(s.prediction * 4) / 4

            // draw approximate fill
            let d = ``

            if (closestQuarter > 0) {
                // vectors from center point to outer points
                const vec0 = [s.outerPoints[0][0] - s.centerPoint[0], s.outerPoints[0][1] - s.centerPoint[1]]
                const vec1 = [s.outerPoints[1][0] - s.centerPoint[0], s.outerPoints[1][1] - s.centerPoint[1]]

                // segment fill points
                const fillPoint0 = [s.centerPoint[0] + closestQuarter * vec0[0], s.centerPoint[1] + closestQuarter * vec0[1]]
                const fillPoint1 = [s.centerPoint[0] + closestQuarter * vec1[0], s.centerPoint[1] + closestQuarter * vec1[1]]

                d += `M${s.centerPoint[0]},${s.centerPoint[1]}`
                d += `L${fillPoint0[0]},${fillPoint0[1]}`
                d += `L${fillPoint1[0]},${fillPoint1[1]}`
                d += `Z`
            }

            return d
        })
        .attr("stroke", "none")
        .attr("fill", s => tableau20[s.classIndex])
}


function drawGlyphSegmentLines(contentGroup, glyphData, strokeWidth) {
    contentGroup.selectAll(".glyph-segment-lines")
        .data(glyphData)
        .enter()
        .append("path")
        .attr("class", "glyph-segment-lines")
        .attr("d", g => {
            // vectors from center point to circle points
            const circlePointsVec = g.circlePoints.map(cp => [cp[0] - g.mx, cp[1] - g.my])

            let d = ``

            // draw three circles of segment lines
            for (let i = 1; i <= 3; i++) {

                d += `M${g.mx + i/4 * circlePointsVec[0][0]},${g.my + i/4 * circlePointsVec[0][1]}`  // move to first point
                for (let j = 1; j < circlePointsVec.length; j++) {
                    d += `L${g.mx + i/4 * circlePointsVec[j][0]},${g.my + i/4 * circlePointsVec[j][1]}`  // lines to other points
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
        .attr("stroke", "black")
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
        .attr("x", d => d.cx)
        .attr("y", d => d.cy)
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


export function drawBinaryGlyphs(contentGroup, xScale, yScale, feature_column) {
    const { glyphSize, glyphBoundingSize } = getGlyphSize(xScale, yScale)
    const { glyphData, segmentData } = calculateGlyphData(xScale, yScale, glyphSize, glyphBoundingSize, feature_column)
    const strokeWidth = getStrokeWidth()

    // removing old glyphs
    clearGlyphs(contentGroup)

    // segment fills
    drawGlyphFillsBinary(contentGroup, segmentData)

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
    drawGlyphFillsSegments(contentGroup, segmentData, strokeWidth)

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
    drawGlyphFillsBinary(contentGroup, segmentData)

    // glyph lines (outline and segment borders)
    drawGlyphLines(contentGroup, glyphData, strokeWidth)

    // whiskers
    drawGlyphWhiskers(contentGroup, segmentData, strokeWidth)

    // details overlay on click
    overlayOnClick(contentGroup, glyphSize, glyphData)
}
