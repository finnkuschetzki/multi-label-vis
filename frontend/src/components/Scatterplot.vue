<script setup>
import axios from "axios"
import { ref, onMounted, watchEffect } from "vue"
import * as d3 from "d3"

// data
const data = ref()

const ax = axios.create({
  baseURL: "http://localhost:5000",
  timeout: 1000
})

ax.get("data/")
    .then(res => {
      data.value = res.data
      console.log(data.value)
    })

// chart
const chart = ref()

function updateChart() {

  const margin = { top: 25, bottom: 25, left: 25, right: 25 }
  const size = 700

  d3.select(chart.value).select("svg").remove()  // removing existing chart

  const svg = d3.select(chart.value).append("svg")
      .attr("width", size)
      .attr("height", size)
      .append("g")

  const outline = svg.append("rect")
      .attr("x", 0)
      .attr("y", 0)
      .attr("width", size)
      .attr("height", size)
      .attr("fill", "none")
      .attr("stroke", "black")
      .attr("stroke-width", 1)

  const xScale = d3.scaleLinear()
      .domain([0,1])
      .range([0, size - margin.left - margin.right])

  const yScale = d3.scaleLinear()
      .domain([0,1])
      .range([0, size - margin.top - margin.right])

  svg.selectAll("circle")
      .data(data.value)
      .enter()
      .append("circle")
      .attr("cx", d => xScale(d["pca_features_or"][0]) + margin.left)
      .attr("cy", d => yScale(d["pca_features_or"][1]) + margin.bottom)
      .attr("r", d => (xScale(0.01) - xScale(0)) / 2)
      .attr("fill", "steelblue")

}

onMounted(updateChart)
watchEffect(updateChart)
</script>

<template>
  <div class="chart-container" ref="container">

    <div id="chart" ref="chart"></div>

  </div>
</template>

<style scoped>
.chart-container {
  flex: 1;
  display: flex;
  justify-content: center;
  align-items: center;
}
</style>