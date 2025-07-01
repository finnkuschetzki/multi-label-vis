<script setup>
import * as d3 from "d3"
import { ref, onMounted, watchEffect } from "vue"

const props = defineProps({
  data: {
    type: Array,
    required: true
  },
  useDGrid: {
    type: Boolean,
    required: true
  },
  dimensionalityReduction: {
    type: String,
    required: true
  },
  highlightClass: {
    type: Number,
    required: true
  }
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

  let feature_column = `${props.dimensionalityReduction}_features`;
  if (props.useDGrid) feature_column += "_or"

  svg.selectAll("circle")
      .data(props.data)
      .enter()
      .append("circle")
      .attr("cx", d => xScale(d[feature_column][0]) + margin.left)
      .attr("cy", d => yScale(d[feature_column][1]) + margin.bottom)
      .attr("r", d => (xScale(0.01) - xScale(0)) / 2)
      .attr("fill", d => {
        if (props.highlightClass === -1) return "steelblue"
        else return d["ground_truth"][props.highlightClass] ? "red" : "steelblue"
      })

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