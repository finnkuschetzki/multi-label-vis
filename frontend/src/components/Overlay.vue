<script setup>
import { onMounted, onUnmounted } from "vue"

import { showOverlay, dataPointGroundTruth, dataPointPredictions } from "@/stores/overlay.js"
import OverlayValueList from "@/components/OverlayValueList.vue";


function hideOverlay() {
  showOverlay.value = false
}

function hideOverlayOnEsc(event) {
  if (event.key === "Escape") {
    console.log("esc")
    hideOverlay()
  }
}

onMounted(() => {
  document.addEventListener("keydown", hideOverlayOnEsc)
})

onUnmounted(() => {
  document.removeEventListener("keydown", hideOverlayOnEsc)
})
</script>

<template>
  <div class="overlay-container">
    <div class="content-container">

      <div class="content-column values-column">
        <OverlayValueList title="Ground Truth" :values="dataPointGroundTruth" />
      </div>

      <div class="content-column values-column">
        <OverlayValueList title="Predictions" :values="dataPointPredictions" :decimal-digits="3" />
      </div>

      <div class="button-column">
        <Button icon="pi pi-times" variant="text" rounded @click="hideOverlay" class="close-button" />
      </div>

    </div>

  </div>
</template>

<style scoped>
.overlay-container {
  border: 2px solid black;
  border-radius: 16px;
  padding: 16px;
  background-color: white;
}

.content-container {
  width: 100%;
  height: 100%;
  display: flex;
  justify-content: space-between;
}

.content-column {
  height: 100%;
  max-height: 100%;
}

.values-column {
  width: 25%;
}

.button-column {
  width: 10%;
}

.close-button {
  float: right
}
</style>