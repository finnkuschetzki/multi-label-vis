<script setup>
import { ref, onMounted, onUnmounted, watch } from "vue"

import { showOverlay, dataPointGroundTruth, dataPointPredictions, dataPointImagePath } from "@/stores/overlay.js"
import OverlayValueList from "@/components/OverlayValueList.vue";
import httpClient from "@/httpClient/httpClient.js";


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


const isImageSet = ref(false)
const imageURL = ref()

watch(dataPointImagePath, async (newDataPointImagePath) => {
  const res = await httpClient.get("image/", {
    params: {
      "imagePath": newDataPointImagePath
    },
    responseType: "blob"
  })

  imageURL.value = URL.createObjectURL(res.data)
  isImageSet.value = true
})
</script>

<template>
  <div class="overlay-container">
    <div class="content-container">

      <div class="content-column image-column">
        <img v-if="isImageSet" :src="imageURL" />
      </div>

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

.image-column {
  width: 40%;
  display: flex;
  justify-content: center;
  align-items: center;
}

.image-column > img {
  max-width: 100%;
  max-height: 100%;
}

.values-column {
  width: 25%;
}

.button-column {
  width: 5%;
}

.close-button {
  float: right
}
</style>