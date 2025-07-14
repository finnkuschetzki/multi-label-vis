import * as d3 from "d3"

import { data } from "@/stores/data.js";
import * as settings from "@/stores/settings.js";


const tableau20 = [
    "#4e79a7", "#8cd17d", "#e15759", "#fabfd2", "#a0cbe8", "#b6992d", "#ff9d9a", "#b07aa1",
    "#f28e2b", "#f1ce63", "#79706e", "#d4a6c8", "#ffbe7d", "#499894", "#bab0ac", "#9d7660",
    "#59a14f", "#86bcb6", "#d37295", "#d7b5a6"
]

const margin = { top: 25, bottom: 25, left: 25, right: 25 }
let svg, xScale, yScale, contentGroup, zoom


export function setupChart(chart, chart_width, chart_height, factorX, factorY) {
    console.log("chart setup")

    // chart base setup
    svg = d3.select(chart.value).append("svg")
        .attr("width", chart_width)
        .attr("height", chart_height)
        .append("g")

    xScale = d3.scaleLinear()
        .domain([0, factorX])
        .range([0, chart_width - margin.left - margin.right])

    yScale = d3.scaleLinear()
        .domain([0, factorY])
        .range([0, chart_height - margin.top - margin.right])

    svg.append("rect")  // bounding rect (outline and events)
        .attr("x", 0)
        .attr("y", 0)
        .attr("width", chart_width)
        .attr("height", chart_height)
        .attr("fill", "none")
        .attr("stroke", "black")
        .attr("stroke-width", 3)
        .attr("pointer-events", "all")

    contentGroup = svg.append("g")  // all chart contents inside

    // zooming
    zoom = d3.zoom()
        .scaleExtent([1, 50])
        .translateExtent([[0, 0], [chart_width, chart_height]])
        .on("zoom", (event) => {
            contentGroup.attr("transform", event.transform)
        })

    svg.call(zoom)
}


export function updateChart() {
    console.log("chart update")

    let feature_column = `${settings.dimensionalityReduction.value}_features`;
    if (settings.useDGrid.value) feature_column += "_or"

    const numClasses = data.value[0]["ground_truth"].length

    const glyphSize = Math.min(
        (xScale(0.01) - xScale(0)),
        (yScale(0.01) - yScale(0))
    ) * 0.9  // todo scale size to prevent overlaps (caused by stroke width)

    // calculating glyphData
    const glyphData = []

    const initialRadians = 3 / 2 * Math.PI
    const classStep = (2 * Math.PI) / numClasses

    data.value.forEach(d => {
        const cx = xScale(d[feature_column][0]) + margin.left
        const cy = yScale(d[feature_column][1]) + margin.bottom

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

    // removing old glyphs
    contentGroup.selectAll(".glyph-lines").remove()
    contentGroup.selectAll(".glyph-segment-fills").remove()

    // segment fills
    contentGroup.selectAll(".glyph-segment-fills")
        .data(segmentData)
        .enter()
        .append("path")
        .attr("class", "glyph-segment-fills")
        .attr("d", s => {
            let d = ``

            // segment fills
            for (let i = 0; i < numClasses; i++) {
                d += `M${s.centerPoint[0]},${s.centerPoint[1]}`
                d += `L${s.outerPoints[0][0]},${s.outerPoints[0][1]}`
                d += `L${s.outerPoints[1][0]},${s.outerPoints[1][1]}`
                d += `Z`
            }

            return d
        })
        .attr("stroke", "none")
        .attr("fill", s => s.binarizedPrediction ? tableau20[s.classIndex] : "none")

    // glyph lines (outline and segment borders)
    contentGroup.selectAll(".glyph-lines")
        .data(glyphData)
        .enter()
        .append("path")
        .attr("class", "glyph-lines")
        .attr("d", g => {
            let d = ``

            // segment borders
            for (let i = 0; i < numClasses; i++) {
                d += `M${g.mx},${g.my}`
                d += `L${g.circlePoints[i][0]},${g.circlePoints[i][1]}`
            }

            // outline
            d += `M${g.circlePoints[0][0]},${g.circlePoints[0][1]}`  // move to first point
            for (let i = 1; i < numClasses; i++) {
                d += `L${g.circlePoints[i][0]},${g.circlePoints[i][1]}`  // lines to other points
            }
            d += `Z`  // complete to first point

            return d
        })
        .attr("stroke", "black")
        .attr("stroke-width", .1)
        .attr("fill", "none")
}
