<script setup>
import Scatterplot from "@/components/Scatterplot.vue"
import { ref } from "vue"

// menu refs
const useDGrid = ref(true)

const dimensionalityReductionOptions = ref([
  { name: "PCA", value: "pca" },
  { name: "UMAP", value: "umap" },
  { name: "t-SNE", value: "tsne" }
])
const dimensionalityReduction = ref("pca")
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
        />
      </div>

    </div>

    <Scatterplot
        :use-d-grid="useDGrid"
        :dimensionality-reduction="dimensionalityReduction"
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
</style>
