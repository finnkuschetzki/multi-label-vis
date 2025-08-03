<script setup>
import Menu from "@/components/Menu.vue"
import Scatterplot from "@/components/Scatterplot.vue"
import Legend from "@/components/Legend.vue"

import { ref, useTemplateRef, onMounted, watch, nextTick } from "vue"


import { showOverlay, overlayPosition, dataPointImagePath, dataPointGroundTruth, dataPointPredictions } from "@/stores/overlay.js"
import * as settings from "@/stores/settings.js"


function resetOverlay() {
  showOverlay.value = false
  overlayPosition.value = null
  dataPointImagePath.value = null
  dataPointGroundTruth.value = null
  dataPointPredictions.value = null
}


const legend = useTemplateRef("legend")
const scatterplot = useTemplateRef("scatterplot")


const displayScatterplot = ref(false)


onMounted(async () => {
  watch(
      [settings.modelName, settings.dataType],
      async () => {
        displayScatterplot.value = false
        resetOverlay()
        await nextTick()

        await legend.value.setup()
        displayScatterplot.value = true
        await nextTick()

        await scatterplot.value.setup()
      },
      { immediate: true }
  )

})
</script>

<template>
  <div class="main-container">

    <div class="sidebar">
      <Menu />
      <Legend ref="legend" />
    </div>

    <Scatterplot v-if="displayScatterplot" ref="scatterplot" />

  </div>
</template>

<style scoped>
.main-container {
  display: flex;
  flex-wrap: wrap;
}

.sidebar {
  margin: .75rem;
}
</style>
