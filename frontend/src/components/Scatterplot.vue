<script setup>
import axios from "axios"
import * as d3 from "d3"
import { ref, onMounted, watch, watchEffect } from "vue"
import { useElementSize } from "@vueuse/core"

import { data } from "@/stores/data.js"
import * as settings from "@/stores/settings.js"


// creating axios instance
const ax = axios.create({
  baseURL: "http://localhost:5000",
  timeout: 1000
})


// sizing and data request
const container = ref()
const { width: containerWidth, height: containerHeight } = useElementSize(container)
let isInitialized = false

const chart = ref()
const chart_width = ref(NaN)
const chart_height = ref(NaN)
const factorX = ref(1)
const factorY = ref(1)

watch([containerWidth, containerHeight], ([newWidth, newHeight]) => {
  if (newWidth === 0 && newHeight === 0) return

  if (!isInitialized) {
    isInitialized = true
    chart_width.value = newWidth * 0.9
    chart_height.value = newHeight * 0.9
    console.log(`initialized width, height to values ${newWidth}, ${newHeight}`)

    factorX.value = chart_width.value / chart_height.value
    factorY.value = 1
    console.log(`set factorX, factorY to values ${factorX.value}, ${factorY.value}`)

    ax.get("data/", {
      params: {
        "factorX": factorX.value,
        "factorY": factorY.value
      }
    }).then(res => {
          data.value = res.data
          console.log(data.value)
        })
  } else {
    console.log("resize")
  }
})


// drawing chart
function drawChart() {
  if (!isNaN(chart_width) && !isNaN(chart_height)) return

  const margin = { top: 25, bottom: 25, left: 25, right: 25 }
  const width = chart_width.value
  const height = chart_height.value

  d3.select(chart.value).select("svg").remove()  // remove existing chart

  const svg = d3.select(chart.value).append("svg")
      .attr("width", width)
      .attr("height", height)
      .append("g")

  const xScale = d3.scaleLinear()
      .domain([0, factorX.value])
      .range([0, width - margin.left - margin.right])

  const yScale = d3.scaleLinear()
      .domain([0, factorY.value])
      .range([0, height - margin.top - margin.right])

  const outline = svg.append("rect")
      .attr("x", 0)
      .attr("y", 0)
      .attr("width", width)
      .attr("height", height)
      .attr("fill", "none")
      .attr("stroke", "black")
      .attr("stroke-width", 3)

  let feature_column = `${settings.dimensionalityReduction.value}_features`;
  if (settings.useDGrid.value) feature_column += "_or"

  svg.selectAll("circle")
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


onMounted(drawChart)
watchEffect(drawChart)
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