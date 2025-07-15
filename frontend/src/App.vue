<script setup>
import Menu from "@/components/Menu.vue"
import Scatterplot from "@/components/Scatterplot.vue"
import Legend from "@/components/Legend.vue"

import { useTemplateRef } from "vue"

/*
Scatterplot computes the space available for the chart (onMounted) to request fitting data from backend.
Therefore, Legend must be rendered with its final size before Scatterplot can be mounted.
 */
const isLegendInitialized = useTemplateRef("isLegendInitialized")
</script>

<template>
  <div class="main-container">

    <Menu />
    <Scatterplot v-if="isLegendInitialized" />
    <Legend ref="isLegendInitialized" />

  </div>
</template>

<style scoped>
.main-container {
  display: flex;
  flex-wrap: wrap;
}
</style>
