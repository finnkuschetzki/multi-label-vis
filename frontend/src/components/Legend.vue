<script setup>
import * as d3 from "d3"

import { ref, useTemplateRef, onMounted, watch } from "vue"

import httpClient from "@/httpClient/httpClient.js"

import { classInfo } from "@/stores/data"
import * as settings from "@/stores/settings"


const tableau20 = [
  "#4e79a7", "#8cd17d", "#e15759", "#fabfd2", "#a0cbe8", "#b6992d", "#ff9d9a", "#b07aa1",
  "#f28e2b", "#f1ce63", "#79706e", "#d4a6c8", "#ffbe7d", "#499894", "#bab0ac", "#9d7660",
  "#59a14f", "#86bcb6", "#d37295", "#d7b5a6"
]


const legendGlyphs = useTemplateRef("legendGlyphs")

function calculateGlyphPoints() {
  const numClasses = classInfo.value.length
  const glyphSize = 14

  const initialRadians = 3 / 2 * Math.PI
  const classStep = (2 * Math.PI) / numClasses

  const mx = 8
  const my = 8

  const circlePoints = []
  for (let i = 0; i < numClasses; i++) {
    circlePoints.push([
      mx + Math.cos(initialRadians + classStep * i) * glyphSize/2,  // x pos
      my + Math.sin(initialRadians + classStep * i) * glyphSize/2  // y pos
    ])
  }

  return { mx, my, circlePoints }
}

function createLegendGlyph(legendGlyph, glyphPoints, i) {
  const numClasses = classInfo.value.length

  // clearing old legendGlyph
  d3.select(legendGlyph).selectAll("*").remove()

  // creating new legendGlyph
  const svg = d3.select(legendGlyph).append("svg")
      .attr("width", 16)
      .attr("height", 16)
      .append("g")

  const { mx, my, circlePoints } = glyphPoints

  // glyph lines
  svg.append("path")
      .attr("d", () => {
        let d = ``

        // segment borders
        for (let i = 0; i < circlePoints.length; i++) {
          d += `M${mx},${my}`
          d += `L${circlePoints[i][0]},${circlePoints[i][1]}`
        }

        // outline
        d += `M${circlePoints[0][0]},${circlePoints[0][1]}`  // move to first point
        for (let i = 1; i < circlePoints.length; i++) {
          d += `L${circlePoints[i][0]},${circlePoints[i][1]}`  // lines to other points
        }
        d += `Z`  // complete to first point

        return d
      })
      .attr("stroke", "black")
      .attr("stroke-width", 0.2)
      .attr("fill", "none")

  // glyph fill
  svg.append("path")
      .attr("d", () => {
        let d = ``

        const circlePoint0 = circlePoints[i]
        const circlePoint1 = circlePoints[(i+1) % numClasses]

        // segment fill
        d += `M${mx},${my}`
        d += `L${circlePoint0[0]},${circlePoint0[1]}`
        d += `L${circlePoint1[0]},${circlePoint1[1]}`
        d += `Z`

        return d
      })
      .attr("stroke", "none")
      .attr("fill", () => tableau20[i])
}


const isInitialized = ref(false)


async function requestClassInfo() {
  const res = await httpClient.get("class-info/", {
    params: {
      "modelName": settings.modelName.value
    }
  })
  classInfo.value = res.data
  console.log(classInfo.value)
}

onMounted(async () => {
  watch(
      settings.modelName,
      async () => {
        await requestClassInfo()

        const glyphPoints = calculateGlyphPoints()
        legendGlyphs.value.forEach((el, i) => createLegendGlyph(el, glyphPoints, i))

        isInitialized.value = true
      },
      { immediate: true }
  )

})


defineExpose(isInitialized)
</script>

<template>
 <div class="legend-container">

  <Card>
    <template #title>Color Legend</template>
    <template #content>

      <div v-for="(c, index) of classInfo" :key="index" class="class-item">

        <span class="legend-glyph" ref="legendGlyphs"></span>
        <span class="color-box" :style="{ backgroundColor: tableau20[index] }"></span>
        <span>{{ c["name"] }}</span>

      </div>

    </template>
  </Card>

 </div>
</template>

<style scoped>
.legend-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  margin-top: 0.75rem;
}

.legend-container > * {
  margin: .5rem .25rem;
}

.class-item {
  display: flex;
  align-items: center;
  gap: 4px;
}

.legend-glyph {
  display: inline-block;
  width: 16px;
  height: 16px;
}

.color-box {
  display: inline-block;
  width: 16px;
  height: 16px;
  border-radius: 4px;
}
</style>