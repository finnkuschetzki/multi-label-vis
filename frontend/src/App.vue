<script setup>
import Menu from "@/components/Menu.vue"
import Scatterplot from "@/components/Scatterplot.vue"
</script>

<template>
  <div class="main-container">

    <Menu />
    <Scatterplot />

  </div>
</template>

<style scoped>
.main-container {
  display: flex;
  flex-wrap: wrap;
}
</style>
