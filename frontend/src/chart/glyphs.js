import config from "../../../config.json"

import { data } from "@/stores/data.js"
import * as settings from "@/stores/settings.js"
import { showOverlay, overlayPosition, dataPointGroundTruth, dataPointPredictions, dataPointImagePath } from "@/stores/overlay.js"
import { margin, glyphSizeMultiplier, tableau20 } from "@/chart/settings.js"



/* CALCULATIONS */


function getNumClasses() {
    return data.value[0]["ground_truth"].length
}


function getGlyphBoundingSize(xScale, yScale) {
    const glyphSize = config.models.find(m => m.name === "base-model")["glyphSize"][settings.dataType.value]
    return Math.min(
        (xScale(glyphSize) - xScale(0)),
        (yScale(glyphSize) - yScale(0))
    )
}


function getGlyphSize(xScale, yScale) {
    return getGlyphBoundingSize(xScale, yScale) * glyphSizeMultiplier
}


function calculateGlyphData(xScale, yScale, featureColumn) {
    const numClasses = getNumClasses()
    const glyphBoundingSize = getGlyphBoundingSize(xScale, yScale)
    const glyphSize = getGlyphSize(xScale, yScale)

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
    contentGroup.selectAll(".glyph-segment-fills").remove()
    contentGroup.selectAll(".glyph-whiskers").remove()
    contentGroup.selectAll(".glyph-event-box").remove()
}


function drawCircles(contentGroup, glyphData, xScale, yScale) {
    const glyphSize = getGlyphSize(xScale, yScale)

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


function drawGlyphLines(contentGroup, glyphData) {
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
        .attr("stroke-width", .1)
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


function drawGlyphWhiskers(contentGroup, segmentData) {
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
        .attr("stroke-width", 0.1)
        .attr("fill", "none")
}



/* EVENTS */


export function overlayOnClick(contentGroup, glyphData, xScale, yScale) {
    const glyphSize = getGlyphSize(xScale, yScale)

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
    const { glyphData } = calculateGlyphData(xScale, yScale, feature_column)

    // removing old glyphs
    clearGlyphs(contentGroup)

    // simple glyphs
    drawCircles(contentGroup, glyphData, xScale, yScale)

    // details overlay on click
    overlayOnClick(contentGroup, glyphData, xScale, yScale)
}


export function drawBinaryGlyphs(contentGroup, xScale, yScale, feature_column) {
    const { glyphData, segmentData } = calculateGlyphData(xScale, yScale, feature_column)

    // removing old glyphs
    clearGlyphs(contentGroup)

    // segment fills
    drawGlyphFillsBinary(contentGroup, segmentData)

    // glyph lines (outline and segment borders)
    drawGlyphLines(contentGroup, glyphData)

    // details overlay on click
    overlayOnClick(contentGroup, glyphData, xScale, yScale)
}


export function drawPartialFillGlyphs(contentGroup, xScale, yScale, feature_column) {
    const { glyphData, segmentData } = calculateGlyphData(xScale, yScale, feature_column)

    // removing old glyphs
    clearGlyphs(contentGroup)

    // segment fills
    drawGlyphFillsPartial(contentGroup, segmentData)

    // glyph lines (outline and segment borders)
    drawGlyphLines(contentGroup, glyphData)

    // details overlay on click
    overlayOnClick(contentGroup, glyphData, xScale, yScale)
}


export function drawWhiskerGlyphs(contentGroup, xScale, yScale, feature_column) {
    const { glyphData, segmentData } = calculateGlyphData(xScale, yScale, feature_column)

    // removing old glyphs
    clearGlyphs(contentGroup)

    // segment fills
    drawGlyphFillsBinary(contentGroup, segmentData)

    // glyph lines (outline and segment borders)
    drawGlyphLines(contentGroup, glyphData)

    // whiskers
    drawGlyphWhiskers(contentGroup, segmentData)

    // details overlay on click
    overlayOnClick(contentGroup, glyphData, xScale, yScale)
}
