<script setup>
import { classInfo } from "@/stores/data.js"


const tableau20 = [
  "#4e79a7", "#8cd17d", "#e15759", "#fabfd2", "#a0cbe8", "#b6992d", "#ff9d9a", "#b07aa1",
  "#f28e2b", "#f1ce63", "#79706e", "#d4a6c8", "#ffbe7d", "#499894", "#bab0ac", "#9d7660",
  "#59a14f", "#86bcb6", "#d37295", "#d7b5a6"
]


const props = defineProps({
  "title": {
    required: true
  },
  "values": {
    required: true
  },
  "decimalDigits": {
    default: 0
  }
})
</script>

<template>
  <div class="values-title">{{ props.title }}</div>

  <div class="values-list">
    <div v-for="(val, i) in props.values">

      <span class="class-item">
        <span class="color-box" :style="{ backgroundColor: tableau20[i] }"></span>
        <span :style="{ color: val >= 0.5 ? 'black' : 'silver' }">{{ classInfo[i].name }}:</span>
      </span>
      <span :style="{ color: val >= 0.5 ? 'black' : 'silver' }">{{ val.toFixed(props.decimalDigits) }}</span>

    </div>

  </div>
</template>

<style scoped>
.values-title {
  font-weight: bold;
  text-align: center;
  margin: 6px;
}

.values-list {
  width: 100%;
  height: calc(100% - 32px);
  max-height: 100%;
  display: flex;
  flex-direction: column;
  flex-wrap: wrap;
  align-content: center;
  column-gap: 1.5em;
  text-align: right;
}

.values-list > * {
  display: flex;
  justify-content: space-between;
  gap: 6px;
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