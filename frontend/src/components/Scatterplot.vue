<script setup>
import Overlay from "@/components/Overlay.vue"

import { ref, useTemplateRef, watch, nextTick } from "vue"
import { useElementSize } from "@vueuse/core"

import httpClient from "@/httpClient/httpClient.js"

import { data, convexHulls } from "@/stores/data.js"
import * as settings from "@/stores/settings.js"
import { showOverlay, overlayPosition } from "@/stores/overlay.js"
import { setupChart, updateChart } from "@/chart/base.js"


// --- default settings ---

showOverlay.value = false


// --- template refs

const container = useTemplateRef("container")
const chart = useTemplateRef("chart")


// --- scatterplot setup ---

const isLoading = ref(true)


async function requestData() {
  console.log("data requested")

  // getting container size and calculating factors
  const { width, height } = useElementSize(container)
  const chart_width = width.value * 0.995
  const chart_height = height.value * 0.995
  const factorX = chart_width / chart_height
  const factorY = 1

  // requesting data
  const res = await httpClient.get("data/", {
    params: {
      "modelName": settings.modelName.value,
      "dataType": settings.dataType.value,
      "factorX": factorX,
      "factorY": factorY
    }
  })

  data.value = res.data["data_points"]
  convexHulls.value = res.data["convex_hulls"]

  console.log("data received")
  console.log(data.value)
  console.log(convexHulls.value)

  return { chart_width, chart_height, factorX, factorY }
}


async function setup() {
  isLoading.value = true
  await nextTick()

  const { chart_width, chart_height, factorX, factorY } = await requestData()

  isLoading.value = false
  await nextTick()

  // initial chart setup
  setupChart(chart, chart_width, chart_height, factorX, factorY)
  updateChart()

  // watch for setting changes that do not need data request
  watch(
      [settings.useDGrid, settings.dimensionalityReduction, settings.glyphType, settings.convexHullIndices, settings.focusIndices],
      () => updateChart(),
  )
}


defineExpose({ setup })
</script>

<template>
  <div class="chart-container" ref="container">

    <div v-if="!isLoading" id="chart" ref="chart"></div>
    <Overlay
        v-if="!isLoading"
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