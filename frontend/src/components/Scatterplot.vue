<script setup>
import Overlay from "@/components/Overlay.vue"

import { ref, onMounted, nextTick, watch } from "vue"
import { useElementSize } from "@vueuse/core"

import httpClient from "@/httpClient/httpClient.js"

import { data } from "@/stores/data.js"
import * as settings from "@/stores/settings.js"
import { showOverlay, overlayPosition } from "@/stores/overlay.js"
import { setupChart, updateChart } from "@/chart/base.js"


// standard settings
showOverlay.value = false


const container = ref()
const chart = ref()

let chart_width, chart_height, factorX, factorY


async function requestData() {
  await nextTick()

  // getting container size and calculating factors
  const { width, height } = useElementSize(container)
  chart_width = width.value * 0.995
  chart_height = height.value * 0.995
  factorX = chart_width / chart_height
  factorY = 1

  // requesting data
  const res = await httpClient.get("data/", {
    params: {
      "modelName": settings.modelName.value,
      "dataType": settings.dataType.value,
      "factorX": factorX,
      "factorY": factorY
    }
  })

  data.value = res.data
  console.log(data.value)

  await nextTick()
}

onMounted(async () => {
  await requestData()

  // chart setup and watch update
  setupChart(chart, chart_width, chart_height, factorX, factorY)
  watch(
      [settings.useDGrid, settings.dimensionalityReduction, settings.glyphType],
      () => updateChart(),
      { immediate: true }
  )
  watch(
      [settings.modelName, settings.dataType],
      async () => { await requestData(); updateChart() },
  )
})
</script>

<template>
  <div class="chart-container" ref="container">

    <div v-if="data" id="chart" ref="chart"></div>
    <Overlay
        v-if="data"
        class="floating-overlay"
        :class="{
          'visible': showOverlay,
          'top-overlay': overlayPosition === 'top',
          'bottom-overlay': overlayPosition === 'bottom',
        }"
    />

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
  margin: .75em;
  position: relative;
}

.floating-overlay {
  position: absolute;
  display: none;
  width: 90%;
  height: 40%;
}

.visible {
  display: block;
}

.top-overlay {
  top: 2em;
}

.bottom-overlay {
  bottom: 2em;
}

.loading-container {
  display: flex;
  align-items: center;
  justify-content: center;
}
</style>