import * as d3 from "d3"

import * as settings from "@/stores/settings.js";
import { margin } from "@/chart/settings.js";
import { drawStandardGlyphs } from "@/chart/glyphs.js";


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

    drawStandardGlyphs(contentGroup, xScale, yScale, feature_column)
}
