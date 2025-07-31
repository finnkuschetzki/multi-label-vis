<script setup>
import { ref } from "vue";

import config from "../../../config.json"

import { data } from "@/stores/data.js"
import * as settings from "@/stores/settings.js"


const visible = ref()

// options
const modelNameOptions = config.models.map(m => {
  return { "label": m.displayName, "value": m.name }
})
const dataTypeOptions = [
  { name: "Training", value: "train" },
  { name: "Validation", value: "val" }
]
const dimensionalityReductionOptions = [
  { name: "PCA", value: "pca" },
  { name: "UMAP", value: "umap" },
  { name: "t-SNE", value: "tsne" }
]
const glyphTypeOptions = [
  { name: "Simple", value: "simple" },
  { name: "Ground Truth", value: "groundTruth" },
  { name: "Binary", value: "binary" },
  { name: "Partial Fill", value: "partialFill" },
  { name: "Segment Fill", value: "segmentFill" },
  { name: "Whisker", value: "whisker" },
  { name: "Comparison", value: "comparison" }
]

// standard settings
data.value = null
settings.modelName.value = modelNameOptions[0].value
settings.dataType.value = "val"
settings.useDGrid.value = true
settings.dimensionalityReduction.value = "pca"
settings.glyphType.value = "simple"
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
    <div class="menu-container">

      <Select v-model="settings.modelName" :options="modelNameOptions" option-label="label" option-value="value" />

      <div>
        <SelectButton
          v-model="settings.dataType"
          :options="dataTypeOptions"
          option-label="name"
          option-value="value"
          :allow-empty="false"
        />
      </div>

      <Divider />

      <div class="toggle-switch">
        <label for="d-grid-toggle">DGrid</label>
        <ToggleSwitch v-model="settings.useDGrid" inputId="d-grid-toggle" />
      </div>

      <div>
        <SelectButton
            v-model="settings.dimensionalityReduction"
            :options="dimensionalityReductionOptions"
            option-label="name"
            option-value="value"
            :allow-empty="false"
        />
      </div>

      <div>
        <SelectButton
          v-model="settings.glyphType"
          :options="glyphTypeOptions"
          option-label="name"
          option-value="value"
          :allow-empty="false"
        />
      </div>

    </div>
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
</style>