<script setup>
import { ref } from "vue";

import { data } from "@/stores/data.js"
import * as settings from "@/stores/settings.js"


// options
const dimensionalityReductionOptions = ref([
  { name: "PCA", value: "pca" },
  { name: "UMAP", value: "umap" },
  { name: "t-SNE", value: "tsne" }
])

// standard settings
settings.useDGrid.value = true
settings.dimensionalityReduction.value = "pca"
settings.highlightClass.value = -1
</script>

<template>
  <div class="menu-container">

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
      <div class="radio-button">
        <RadioButton v-model="settings.highlightClass" :input-id="-1" :value="-1" />
        <label :for="-1">Keine Klasse</label>
      </div>
      <div class="radio-button" v-for="(_, index) in data[0]['ground_truth']" :key="index">
        <RadioButton v-model="settings.highlightClass" :input-id="index" :value="index" />
        <label :for="index">Klasse {{ index }}</label>
      </div>
    </div>

  </div>
</template>

<style scoped>
.menu-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  margin-top: 0.75rem;
  margin-bottom: 0.75rem;
  border-right: 2px solid black;
}

.menu-container > * {
  margin: 0.75rem 1.5rem;
}

.toggle-switch {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.radio-button {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin: 0.5rem 0;
}
</style>