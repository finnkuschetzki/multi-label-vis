<script setup>
import axios from "axios"
import * as d3 from "d3"
import { ref, onMounted, nextTick, watchEffect } from "vue"
import { useElementSize } from "@vueuse/core"

import { data } from "@/stores/data.js"
import * as settings from "@/stores/settings.js"


const ax = axios.create({
  baseURL: "http://localhost:5000",
  timeout: 1000
})


const container = ref()
const chart = ref()

let chart_width, chart_height, factorX, factorY

const margin = { top: 25, bottom: 25, left: 25, right: 25 }
let svg, xScale, yScale, contentGroup, zoom


onMounted(async () => {
  await nextTick()

  // getting container size and calculating factors
  const { width, height } = useElementSize(container)
  chart_width = width.value * 0.9
  chart_height = height.value * 0.9
  factorX = chart_width / chart_height
  factorY = 1

  // requesting data
  const res = await ax.get("data/", {
    params: {
      "factorX": factorX,
      "factorY": factorY
    }
  })

  data.value = res.data
  console.log(data.value)

  await nextTick()

  // chart setup and watch update
  setupChart()
  watchEffect(updateChart)
})


function setupChart() {
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
      .scaleExtent([1, 10])
      .translateExtent([[0, 0], [chart_width, chart_height]])
      .on("zoom", (event) => {
        contentGroup.attr("transform", event.transform)
      })

  svg.call(zoom)
}


function updateChart() {
  console.log("chart update")

  let feature_column = `${settings.dimensionalityReduction.value}_features`;
  if (settings.useDGrid.value) feature_column += "_or"

  // removing old glyphs
  contentGroup.selectAll("circle").remove()

  // adding new glyphs
  contentGroup.selectAll("circle")
      .data(data.value)
      .enter()
      .append("circle")
      .attr("cx", d => xScale(d[feature_column][0]) + margin.left)
      .attr("cy", d => yScale(d[feature_column][1]) + margin.bottom)
      .attr("r", Math.min(
          (xScale(0.01) - xScale(0)) / 2,
          (yScale(0.01) - yScale(0)) / 2
      ))
      .attr("fill", d => {
        if (settings.highlightClass.value === -1) return "steelblue"
        else return d["ground_truth"][settings.highlightClass.value] ? "red" : "steelblue"
      })
}
</script>

<template>
  <div class="chart-container" ref="container">

    <div v-if="data" id="chart" ref="chart"></div>

    <div v-else class="loading-container">
      <ProgressSpinner animation-duration=".5s" />
    </div>

  </div>
</template>

<style scoped>
.chart-container {
  flex: 1;
  display: flex;
  justify-content: center;
  align-items: center;
}

.loading-container {
  display: flex;
  align-items: center;
  justify-content: center;
}
</style>