<script setup>
import axios from "axios"
import * as d3 from "d3"
import { ref, onMounted, watchEffect } from "vue"
import { useElementSize } from "@vueuse/core"

import { data } from "@/stores/data.js"
import * as settings from "@/stores/settings.js"


const ax = axios.create({
  baseURL: "http://localhost:5000",
  timeout: 1000
})

ax.get("data/")
    .then(res => {
      data.value = res.data
      console.log(data.value)
    })


const container = ref()
const chart = ref()

const { width, height } = useElementSize(container)


function drawChart() {
  const margin = { top: 25, bottom: 25, left: 25, right: 25 }
  const size = Math.min(width.value, height.value) * 0.9

  d3.select(chart.value).select("svg").remove()  // remove existing chart

  const svg = d3.select(chart.value).append("svg")
      .attr("width", size)
      .attr("height", size)
      .append("g")

  const xScale = d3.scaleLinear()
      .domain([0,1])
      .range([0, size - margin.left - margin.right])

  const yScale = d3.scaleLinear()
      .domain([0,1])
      .range([0, size - margin.top - margin.right])

  const outline = svg.append("rect")
      .attr("x", 0)
      .attr("y", 0)
      .attr("width", size)
      .attr("height", size)
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
      .attr("r", d => Math.min(
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