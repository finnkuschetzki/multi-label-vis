import { data } from "@/stores/data.js"
import { margin, glyphSizeMultiplier, tableau20 } from "@/chart/settings.js"



/* CALCULATIONS */


function getNumClasses() {
    return data.value[0]["ground_truth"].length
}


function getGlyphSize(xScale, yScale) {
    return Math.min(
        (xScale(0.01) - xScale(0)),
        (yScale(0.01) - yScale(0))
    ) * glyphSizeMultiplier
}


function calculateGlyphData(xScale, yScale, featureColumn) {
    const numClasses = getNumClasses()
    const glyphSize = getGlyphSize(xScale, yScale)

    const glyphData = []

    const initialRadians = 3 / 2 * Math.PI
    const classStep = (2 * Math.PI) / numClasses

    data.value.forEach(d => {
        const cx = xScale(d[featureColumn][0]) + margin.left
        const cy = yScale(d[featureColumn][1]) + margin.bottom

        const mx = cx + glyphSize/2
        const my = cy + glyphSize/2

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
            "mx": mx,
            "my": my,
            "circlePoints": circlePoints,
            "segments": segments
        })
    })

    const segmentData = glyphData.flatMap(glyph => glyph.segments)

    return { glyphData, segmentData }
}



/* PARTS OF GLYPHS */


function clearGlyphs(contentGroup) {
    contentGroup.selectAll(".glyph-lines").remove()
    contentGroup.selectAll(".glyph-segment-fills").remove()
    contentGroup.selectAll(".glyph-whiskers").remove()
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



/* FULL GLYPHS */


export function drawBinaryGlyphs(contentGroup, xScale, yScale, feature_column) {
    const { glyphData, segmentData } = calculateGlyphData(xScale, yScale, feature_column)

    // removing old glyphs
    clearGlyphs(contentGroup)

    // segment fills
    drawGlyphFillsBinary(contentGroup, segmentData)

    // glyph lines (outline and segment borders)
    drawGlyphLines(contentGroup, glyphData)
}


export function drawPartialFillGlyphs(contentGroup, xScale, yScale, feature_column) {
    const { glyphData, segmentData } = calculateGlyphData(xScale, yScale, feature_column)

    // removing old glyphs
    clearGlyphs(contentGroup)

    // segment fills
    drawGlyphFillsPartial(contentGroup, segmentData)

    // glyph lines (outline and segment borders)
    drawGlyphLines(contentGroup, glyphData)
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
}
