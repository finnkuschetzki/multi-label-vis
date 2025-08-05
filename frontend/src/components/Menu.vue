<script setup>
import { ref, computed, watch } from "vue"

import config from "../../../config.json"

import { classInfo, data } from "@/stores/data.js"
import * as settings from "@/stores/settings.js"


const tableau20 = [
  "#4e79a7", "#8cd17d", "#e15759", "#fabfd2", "#a0cbe8", "#b6992d", "#ff9d9a", "#b07aa1",
  "#f28e2b", "#f1ce63", "#79706e", "#d4a6c8", "#ffbe7d", "#499894", "#bab0ac", "#9d7660",
  "#59a14f", "#86bcb6", "#d37295", "#d7b5a6"
]


const visible = ref()

// options
const modelNameOptions = config.models.map(m => {
  return { label: m.displayName, value: m.name }
})
const dataTypeOptions = [
  { label: "Training", value: "train" },
  { label: "Validation", value: "val" }
]
const glyphDataOptions = [
  { label: "Simple", value: "simple" },
  { label: "Ground Truth", value: "groundTruth" },
  { label: "Predictions", value: "predictions" },
  { label: "Comparison", value: "comparison" }
]
const dimensionalityReductionOptions = [
  { label: "PCA", value: "pca" },
  { label: "UMAP", value: "umap" },
  { label: "t-SNE", value: "tsne" }
]
const glyphTypeOptions = computed(() => {
  switch (settings.glyphData.value) {
    case "simple":
      return [
        { label: "Simple", value: "simple" }
      ]
    case "groundTruth":
      return [
        { label: "Ground Truth", value: "groundTruth" }
      ]
    case "predictions":
      return [
        { label: "Binary", value: "binary" },
        { label: "Partial Fill", value: "partialFill" },
        { label: "Segment Fill", value: "segmentFill" },
        { label: "Whisker", value: "whisker" }
      ]
    case "comparison":
      return [
        { label: "Comparison", value: "comparison" }
      ]
  }
})

watch(glyphTypeOptions, (newGlyphTypeOptions) => settings.glyphType.value = newGlyphTypeOptions[0].value)

// standard settings
data.value = null
settings.modelName.value = modelNameOptions[0].value
settings.dataType.value = "val"
settings.glyphData.value = "simple"
settings.useDGrid.value = true
settings.dimensionalityReduction.value = "pca"
settings.glyphType.value = "simple"


async function resetIndices() {
  settings.convexHullIndices.value = []
  settings.focusIndices.value = []
  // options will refresh automatically on classInfo change
}


defineExpose({ resetIndices })
</script>

<template>

  <div class="button-container">
    <Button icon="pi pi-bars" class="menu-button" @click="visible = true" />
  </div>

  <Drawer v-model:visible="visible" position="left" header="Settings"
      :pt="{
        root: {
          style: { width: 'auto', maxWidth: 'none' }
        }
      }"
  >
    <ScrollPanel>
      <div class="menu-container">

        <Select v-model="settings.modelName.value" :options="modelNameOptions" option-label="label" option-value="value" />

        <div>
          <SelectButton
            v-model="settings.dataType.value"
            :options="dataTypeOptions"
            option-label="label"
            option-value="value"
            :allow-empty="false"
          />
        </div>

        <div>
          <SelectButton
              v-model="settings.glyphData.value"
              :options="glyphDataOptions"
              option-label="label"
              option-value="value"
              :allow-empty="false"
          />
        </div>

        <Divider />

        <div class="toggle-switch">
          <label for="d-grid-toggle">DGrid</label>
          <ToggleSwitch v-model="settings.useDGrid.value" inputId="d-grid-toggle" />
        </div>

        <div>
          <SelectButton
              v-model="settings.dimensionalityReduction.value"
              :options="dimensionalityReductionOptions"
              option-label="label"
              option-value="value"
              :allow-empty="false"
          />
        </div>

        <div>
          <SelectButton
              v-model="settings.glyphType.value"
              :options="glyphTypeOptions"
              option-label="label"
              option-value="value"
              :allow-empty="false"
          />
        </div>

        <Divider />

        <div v-if="classInfo && (settings.glyphData.value === 'groundTruth' || settings.glyphData.value === 'predictions')">
          <div v-for="(c, index) in classInfo" class="class-item" :key="index">
            <Checkbox v-model="settings.convexHullIndices.value" size="small" :value="index" />
            <span class="color-box" :style="{ backgroundColor: tableau20[index] }"></span>
            <span>{{ c["name"] }}</span>
          </div>
        </div>

        <Divider v-if="classInfo && (settings.glyphData.value === 'groundTruth' || settings.glyphData.value === 'predictions')" />

        <div v-if="classInfo && (settings.glyphData.value === 'groundTruth' || settings.glyphData.value === 'predictions')">
          <div v-for="(c, index) in classInfo" class="class-item" :key="index">
            <Checkbox v-model="settings.focusIndices.value" size="small" :value="index" />
            <span class="color-box" :style="{ backgroundColor: tableau20[index] }"></span>
            <span>{{ c["name"] }}</span>
          </div>
        </div>

      </div>
    </ScrollPanel>
  </Drawer>
</template>

<style scoped>
.button-container {
  display: flex;
}

.button-container > Button {
  flex: 1;
}

.menu-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  margin-top: 0.75rem;
  margin-bottom: 0.75rem;
}

.menu-container > * {
  margin: 0.75rem 1.5rem;
}

.toggle-switch {
  display: flex;
  align-items: center;
  gap: 0.5rem;
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