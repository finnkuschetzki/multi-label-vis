<script setup lang="ts">
import { ref, onMounted } from "vue"

import httpClient from "@/httpClient/httpClient.js"

import { classInfo } from '@/stores/data'


const tableau20 = [
  "#4e79a7", "#8cd17d", "#e15759", "#fabfd2", "#a0cbe8", "#b6992d", "#ff9d9a", "#b07aa1",
  "#f28e2b", "#f1ce63", "#79706e", "#d4a6c8", "#ffbe7d", "#499894", "#bab0ac", "#9d7660",
  "#59a14f", "#86bcb6", "#d37295", "#d7b5a6"
]

const isInitialized = ref(false)


onMounted(async () => {
  const res = await httpClient.get("class-info/", {
    params: {
      "modelName": "base-model"
    }
  })
  classInfo.value = res.data

  isInitialized.value = true
})


defineExpose(isInitialized)
</script>

<template>
 <div class="legend-container">

  <Card>
    <template #title>Color Legend</template>
    <template #content>

      <div v-for="(c, index) of classInfo" class="class-item">

        <span class="color-box" :style="{ backgroundColor: tableau20[index] }"></span>
        <span>{{ c["name"] }}</span>

      </div>

    </template>
  </Card>

 </div>
</template>

<style scoped>
.legend-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  margin-top: 0.75rem;
}

.legend-container > * {
  margin: .5rem .25rem;
}

.class-item {
  display: flex;
  align-items: center;
  gap: 4px;
}

.color-box {
  display: inline-block;
  width: 16px;
  height: 16px;
  border-radius: 4px;
}
</style>