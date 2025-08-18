<script setup>
import { ref } from "vue"

const props = defineProps({
  "header": {
    type: String,
    required: true
  }
})

const dialogVisible = ref()

function openDialog() {
  dialogVisible.value = true
}
</script>

<template>
  <Button icon="pi pi-info-circle" size="normal" severity="secondary" variant="text" rounded @click="openDialog"/>

  <Dialog v-model:visible="dialogVisible" :header="props.header" position="topright" :draggable="false" :style="{ width: '40%' }">
    <slot></slot>
  </Dialog>
</template>

<style scoped>

</style>