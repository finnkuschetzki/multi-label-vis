<script setup>
import { ref } from "vue"

// menu refs
const useDGrid = ref(true)
</script>

<template>
  <div class="main-container">

    <div class="menu-container">

      <div class="toggle-switch-container">
        <label for="d-grid-toggle">DGrid</label>
        <ToggleSwitch v-model="useDGrid" inputId="d-grid-toggle" />
      </div>

    </div>

  </div>
</template>

<style scoped>
.main-container {
  display: flex;
  flex-wrap: wrap;
}

.menu-container {
  display: inline-block;
  padding: 1.5rem;
  border-right: 3px solid lightgrey;
}

.toggle-switch-container {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}
</style>
