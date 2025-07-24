<script setup>
import { showOverlay } from "@/stores/overlay.js"
import { onMounted, onUnmounted } from "vue";

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

    <Button icon="pi pi-times" variant="text" rounded @click="hideOverlay" class="close-button" />

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

.close-button {
  float: right
}
</style>