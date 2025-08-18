<script setup>
import InfoItem from "@/components/InfoItem.vue";

import { ref, computed, watch } from "vue"

import config from "../../../config.json"

import { classInfo, data } from "@/stores/data.js"
import * as settings from "@/stores/settings.js"


const tableau20 = [
  "#4e79a7", "#8cd17d", "#e15759", "#fabfd2", "#a0cbe8", "#b6992d", "#ff9d9a", "#b07aa1",
  "#f28e2b", "#f1ce63", "#79706e", "#d4a6c8", "#ffbe7d", "#499894", "#bab0ac", "#9d7660",
  "#59a14f", "#86bcb6", "#d37295", "#d7b5a6"
]


const visible = ref()

// options
const modelNameOptions = config.models.map(m => {
  return { label: m.displayName, value: m.name }
})
const dataTypeOptions = [
  { label: "Training", value: "train" },
  { label: "Validation", value: "val" }
]
const glyphDataOptions = [
  { label: "Simple", value: "simple" },
  { label: "Ground Truth", value: "groundTruth" },
  { label: "Predictions", value: "predictions" },
  { label: "Comparison", value: "comparison" }
]
const dimensionalityReductionOptions = [
  { label: "PCA", value: "pca" },
  { label: "UMAP", value: "umap" },
  { label: "t-SNE", value: "tsne" }
]
const glyphTypeOptions = computed(() => {
  switch (settings.glyphData.value) {
    case "simple":
      return [
        { label: "Simple", value: "simple" }
      ]
    case "groundTruth":
      return [
        { label: "Ground Truth", value: "groundTruth" }
      ]
    case "predictions":
      return [
        { label: "Binary", value: "binary" },
        { label: "Partial Fill", value: "partialFill" },
        { label: "Segment Fill", value: "segmentFill" },
        { label: "Whisker", value: "whisker" }
      ]
    case "comparison":
      return [
        { label: "Binary", value: "comparison-binary" },
        { label: "Opacity", value: "comparison-opacity" }
      ]
  }
})
const focusSetOperationOptions = [
  { label: "Union", value: "union" },
  { label: "Intersection", value: "intersection" }
]

watch(glyphTypeOptions, (newGlyphTypeOptions) => settings.glyphType.value = newGlyphTypeOptions[0].value)

// standard settings
data.value = null
settings.modelName.value = modelNameOptions[0].value
settings.dataType.value = "val"
settings.glyphData.value = "simple"
settings.useDGrid.value = true
settings.dimensionalityReduction.value = "pca"
settings.glyphType.value = "simple"
settings.focusSetOperation.value = "union"


async function resetIndices() {
  settings.convexHullIndices.value = []
  settings.focusIndices.value = []
  // options will refresh automatically on classInfo change
}


defineExpose({ resetIndices })
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
    <ScrollPanel>
      <div class="menu-container">

        <div>
          <Select v-model="settings.modelName.value" :options="modelNameOptions" option-label="label" option-value="value" />
          <InfoItem header="Dataset">
            <p>
              Datasets are subsets from the COCO dataset.
            </p>
            <ul>
              <li><i>Electronic, Furniture, Kitchen</i>: all classes from superclasses "electronic", "furniture", "kitchen"</li>
              <li><i>Person, Sports, Vehicle</i>: all classes from superclasses "person", "sports", "vehicle"</li>
              <li><i>Top 20 classes</i>: top 20 classes from the COCO dataset with the highest number of images</li>
            </ul>
          </InfoItem>
        </div>

        <div>
          <SelectButton
            v-model="settings.dataType.value"
            :options="dataTypeOptions"
            option-label="label"
            option-value="value"
            :allow-empty="false"
          />
          <InfoItem header="Data Partition">
            <ul>
              <li><i>Training</i>: contains all data points from training data</li>
              <li><i>Validation</i>: contains all data points from validation data</li>
            </ul>
          </InfoItem>
        </div>

        <div>
          <SelectButton
              v-model="settings.glyphData.value"
              :options="glyphDataOptions"
              option-label="label"
              option-value="value"
              :allow-empty="false"
          />
          <InfoItem header="Data Type">
            <ul>
              <li><i>Simple</i>: shows only data points positions</li>
              <li><i>Ground Truth</i>: encodes ground truth labels in glyphs</li>
              <li><i>Predictions</i>: encodes predictions from the image classifier in glyphs</li>
              <li><i>Comparison</i>: encodes correct classifications and misclassifications in glyphs</li>
            </ul>
          </InfoItem>
        </div>

        <Divider />

        <div class="toggle-switch-container">
          <label for="d-grid-toggle">DGrid</label>
          <ToggleSwitch v-model="settings.useDGrid.value" inputId="d-grid-toggle" class="toggle-switch" />
          <InfoItem header="Overlap Removal">
            <p>
              Toggles whether overlap removal using DGrid algorithm is applied.
            </p>
          </InfoItem>
        </div>

        <div>
          <SelectButton
              v-model="settings.dimensionalityReduction.value"
              :options="dimensionalityReductionOptions"
              option-label="label"
              option-value="value"
              :allow-empty="false"
          />
          <InfoItem header="Dimensionality Reduction">
            <p>
              Data points features are extracted from the image classifiers feature layer (last dense layer).
              These features are dimensionally-reduced using the selected dimensionality reduction technique.
              Resulting two-dimensional embeddings are used as position for the data point.
            </p>
            <ul>
              <li><i>PCA</i>: Principal Component Analysis (linear, global)</li>
              <li><i>UMAP</i>: Uniform Manifold Approximation and Projection (non-linear, local)</li>
              <li><i>t-SNE</i>: t-Distributed Stochastic Neighbor Embedding (non-linear, local)</li>
            </ul>
          </InfoItem>
        </div>

        <div>
          <SelectButton
              v-model="settings.glyphType.value"
              :options="glyphTypeOptions"
              option-label="label"
              option-value="value"
              :allow-empty="false"
          />
          <InfoItem header="Glyph Type">
            <div v-if="settings.glyphData.value === 'simple'">
              <ul>
                <li><i>Simple</i>: circular glyph</li>
              </ul>
            </div>
            <div v-else-if="settings.glyphData.value === 'groundTruth'">
              <p>
                Glyphs are composed from sectors. Each sector is associated with a label, encoded by sector position and color.
              </p>
              <ul>
                <li><i>Ground Truth</i>: sectors are filled, if associated labels are in ground truth</li>
              </ul>
            </div>
            <div v-else-if="settings.glyphData.value === 'predictions'">
              <p>
                Glyphs are composed from sectors. Each sector is associated with a label, encoded by sector position and color.
              </p>
              <ul>
                <li><i>Binary</i>: sectors are filled, if associated labels are in predictions</li>
                <li><i>Partial Fill</i>: sectors are filled linearly along the sectors middle line,
                  according to their predicted probability (full fill corresponds to probability 1.0, empty fill
                  corresponds to probability 0.0, note: labels count as predicted if their probability is greater
                  than 0.5), all sectors are shown in their associated color</li>
                <li><i>Segment Fill</i>: sectors are divided into five segments, segments are filled to the approximate predicted probability</li>
              </ul>
            </div>
            <div v-else-if="settings.glyphData.value === 'comparison'">
              <p>
                Glyphs are composed from sectors. Each sector is associated with a label, encoded by sector position.
                Sector color is gray if label is in both ground truth and prediction, red if label is only ground truth, blue if label is only in prediction.
              </p>
              <ul>
                <li><i>Binary</i>: sectors are filled, color is assigned as described above</li>
                <li><i>Opacity</i>: if a sectors associated is a misclassification (either red or blue color), the
                  sectors opacity maps to the error from a correct classification result, high opacity corresponds to
                  high error (example: a sector whose label has a probability of 0.45 (not predicted) but is in ground
                  truth has a low opacity, corresponding to a low error; a sector whose label has a probability of 0.1
                  (not predicted) but is in ground truth has a high opacity, corresponding to a high error)</li>
              </ul>
            </div>
          </InfoItem>
        </div>

        <Divider />

        <div v-if="classInfo && (settings.glyphData.value === 'groundTruth' || settings.glyphData.value === 'predictions')" class="column-flex">
          <div>
            <div class="title-row">
              <div class="list-title">Convex Hull</div>
              <InfoItem header="Convex Hull">
                <p>
                  For each of the selected labels, the convex hull enclosing all data points associated with that label
                  (either in ground truth or predictions, as selected) is shown. Convex hulls are shown in the labels associated color.
                </p>
              </InfoItem>
            </div>
            <div v-for="(c, index) in classInfo" class="class-item" :key="index">
              <Checkbox v-model="settings.convexHullIndices.value" size="small" :value="index" />
              <span class="color-box" :style="{ backgroundColor: tableau20[index] }"></span>
              <span>{{ c["name"] }}</span>
            </div>
          </div>

          <Divider layout="vertical" />

          <div>

            <div class="title-row">
              <div class="list-title">Focus</div>
              <InfoItem header="Focus on data points">
                <p>
                  Data points associated with the selected labels are focused. Choose between:
                </p>
                <ul>
                  <li><i>union</i> (focus data points associated with at least one selected label)</li>
                  <li><i>intersection</i> (focus data points associated with all selected labels)</li>
                </ul>
                <p>Select these options below the list of labels.</p>
              </InfoItem>
            </div>
            <div v-for="(c, index) in classInfo" class="class-item" :key="index">
              <Checkbox v-model="settings.focusIndices.value" size="small" :value="index" />
              <span class="color-box" :style="{ backgroundColor: tableau20[index] }"></span>
              <span>{{ c["name"] }}</span>
            </div>

            <div class="set-operation-container">
              <div class="set-operation-title">Set Operation</div>
              <div v-for="option in focusSetOperationOptions" class="class-item">
                <RadioButton v-model="settings.focusSetOperation.value" :value="option.value" size="small" />
                <label>{{ option.label }}</label>
              </div>
            </div>

          </div>
        </div>

      </div>
    </ScrollPanel>
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

li {
  margin: .1rem;
}

.toggle-switch-container {
  display: flex;
  align-items: center;
}

.toggle-switch {
  margin-left: 0.5rem;
}

.class-item {
  display: flex;
  align-items: center;
  gap: 4px;
}

.color-box {
  display: inline-block;
  width: 16px;
  height: 16px;
  border-radius: 4px;
}

.column-flex {
  display: flex;
  width: 100%;
  justify-content: space-around;
}

.title-row {
  display: flex;
  align-items: center;
}

.list-title {
  font-weight: bold;
}

.set-operation-container {
  margin-top: 0.75rem;
}

.set-operation-title {
  font-style: italic;
}
</style>