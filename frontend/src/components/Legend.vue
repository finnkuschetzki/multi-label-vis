<script setup>
import * as d3 from "d3"

import { useTemplateRef } from "vue"

import httpClient from "@/httpClient/httpClient.js"

import { classInfo } from "@/stores/data"
import * as settings from "@/stores/settings"


const tableau20 = [
  "#4e79a7", "#8cd17d", "#e15759", "#fabfd2", "#a0cbe8", "#b6992d", "#ff9d9a", "#b07aa1",
  "#f28e2b", "#f1ce63", "#79706e", "#d4a6c8", "#ffbe7d", "#499894", "#bab0ac", "#9d7660",
  "#59a14f", "#86bcb6", "#d37295", "#d7b5a6"
]


// --- template refs ---

const legendGlyphs = useTemplateRef("legendGlyphs")


// --- legend glyphs ---

function calculateGlyphPoints() {
  const numClasses = classInfo.value.length
  const glyphSize = 14

  const initialRadians = 3 / 2 * Math.PI
  const classStep = (2 * Math.PI) / numClasses

  const cx = 8
  const cy = 8

  const circlePoints = []
  for (let i = 0; i < numClasses; i++) {
    circlePoints.push([
      cx + Math.cos(initialRadians + classStep * i) * glyphSize/2,  // x pos
      cy + Math.sin(initialRadians + classStep * i) * glyphSize/2  // y pos
    ])
  }

  return { cx, cy, circlePoints }
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

  const { cx, cy, circlePoints } = glyphPoints

  // glyph lines
  svg.append("path")
      .attr("d", () => {
        let d = ``

        // segment borders
        for (let i = 0; i < circlePoints.length; i++) {
          d += `M${cx},${cy}`
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
        d += `M${cx},${cy}`
        d += `L${circlePoint0[0]},${circlePoint0[1]}`
        d += `L${circlePoint1[0]},${circlePoint1[1]}`
        d += `Z`

        return d
      })
      .attr("stroke", "none")
      .attr("fill", () => tableau20[i])
}


// --- legend setup ---

async function requestClassInfo() {
  console.log("class info requested")

  const res = await httpClient.get("class-info/", {
    params: {
      "modelName": settings.modelName.value
    }
  })

  classInfo.value = res.data

  console.log("class info received")
  console.log(classInfo.value)
}


async function setup() {
  await requestClassInfo()

  const glyphPoints = calculateGlyphPoints()
  legendGlyphs.value.forEach((el, i) => createLegendGlyph(el, glyphPoints, i))
}


defineExpose({ setup })
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

  <Card v-if="settings.glyphData.value === 'comparison'">
    <template #content>

      <div class="misclassification-item">
        <div class="opacity-bar" :class="{
          'red-static': settings.glyphType.value === 'comparison-binary',
          'red-gradient': settings.glyphType.value === 'comparison-opacity'
         }"></div>
        <div>ground truth</div>
        <div>not prediction</div>
      </div>

      <div class="misclassification-item">
        <div class="opacity-bar gray-static"></div>
        <div>ground truth</div>
        <div>prediction</div>
      </div>

      <div class="misclassification-item">
        <div class="opacity-bar" :class="{
          'blue-static': settings.glyphType.value === 'comparison-binary',
          'blue-gradient': settings.glyphType.value === 'comparison-opacity'
         }"></div>
        <div>not ground truth</div>
        <div>prediction</div>
      </div>

    </template>
  </Card>

 </div>
</template>

<style scoped>
.legend-container {
  display: flex;
  flex-direction: column;
  align-items: stretch;
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

.misclassification-item {
  display: flex;
  flex-direction: column;
  align-items: center;
}

.misclassification-item:not(:last-child) {
  margin-bottom: 12px;
}

.opacity-bar {
  height: 16px;
  width: 100%;
  margin-bottom: 4px;
}

.red-static {
  background-color: red;
}

.red-gradient {
  background: linear-gradient(
    to right,
    rgba(255, 0, 0, 0),
    rgba(255, 0, 0, 255)
  );
}

.gray-static {
  background-color: darkgray;
}

.blue-static {
  background-color: dodgerblue;
}

.blue-gradient {
  background: linear-gradient(
    to right,
    rgba(30, 143, 255, 0),
    rgba(30, 143, 255, 255)
  );
}
</style>