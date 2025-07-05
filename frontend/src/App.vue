<script setup>
import axios from "axios"

import Menu from "@/components/Menu.vue"
import Scatterplot from "@/components/Scatterplot.vue"

import { data } from "@/stores/data.js"


const ax = axios.create({
  baseURL: "http://localhost:5000",
  timeout: 1000
})

ax.get("data/")
    .then(res => {
      data.value = res.data
      console.log(data.value)
    })
</script>

<template>
  <div v-if="data" class="main-container">

    <Menu />
    <Scatterplot />

  </div>
  <div v-else-if="false" class="main-container loading">

    <ProgressSpinner animation-duration=".5s" />

  </div>
</template>

<style scoped>
.main-container {
  display: flex;
  flex-wrap: wrap;
}

.loading {
  align-items: center;
  justify-content: center;
}
</style>
