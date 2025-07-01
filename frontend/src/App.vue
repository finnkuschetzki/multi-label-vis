<script setup>
import axios from "axios"
import Scatterplot from "@/components/Scatterplot.vue"
import { ref } from "vue"

// data
const data = ref()

const ax = axios.create({
  baseURL: "http://localhost:5000",
  timeout: 1000
})

ax.get("data/")
    .then(res => {
      data.value = res.data
      console.log(data.value)
    })

// menu refs
const useDGrid = ref(true)

const dimensionalityReductionOptions = ref([
  { name: "PCA", value: "pca" },
  { name: "UMAP", value: "umap" },
  { name: "t-SNE", value: "tsne" }
])
const dimensionalityReduction = ref("pca")

const highlightClass = ref(-1)
</script>

<template>
  <div class="main-container">

    <div class="menu-container">

      <div class="toggle-switch">
        <label for="d-grid-toggle">DGrid</label>
        <ToggleSwitch v-model="useDGrid" inputId="d-grid-toggle" />
      </div>

      <div>
        <SelectButton
            v-model="dimensionalityReduction"
            :options="dimensionalityReductionOptions"
            option-label="name"
            option-value="value"
            :allow-empty="false"
        />
      </div>

      <div>
        <div class="radio-button">
          <RadioButton v-model="highlightClass" :input-id="-1" :value="-1" />
          <label :for="-1">Keine Klasse</label>
        </div>
        <div class="radio-button" v-for="(item, index) in data[0]['ground_truth']" :key="index">
          <RadioButton v-model="highlightClass" :input-id="index" :value="index" />
          <label :for="index">Klasse {{ index }}</label>
        </div>
      </div>

    </div>

    <Scatterplot
        :data="data"
        :use-d-grid="useDGrid"
        :dimensionality-reduction="dimensionalityReduction"
        :highlight-class="highlightClass"
    />

  </div>
</template>

<style scoped>
.main-container {
  display: flex;
  flex-wrap: wrap;
}

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
