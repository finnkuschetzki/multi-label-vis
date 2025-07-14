<script setup>
import axios from "axios"
import { ref, onMounted, nextTick, watchEffect } from "vue"
import { useElementSize } from "@vueuse/core"

import { data, classInfo } from "@/stores/data.js"

import { setupChart, updateChart } from "@/chart/base.js"


const ax = axios.create({
  baseURL: "http://localhost:5000",
  timeout: 1000
})


const container = ref()
const chart = ref()

let chart_width, chart_height, factorX, factorY


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

  data.value = res.data["data_points"]
  classInfo.value = res.data["class_info"]
  console.log(data.value)

  await nextTick()

  // chart setup and watch update
  setupChart(chart, chart_width, chart_height, factorX, factorY)
  watchEffect(updateChart)
})
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