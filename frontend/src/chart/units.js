import config from "../../../config.json";
import * as settings from "@/stores/settings.js";
import { data } from "@/stores/data.js";
import { glyphSizeMultiplier } from "@/chart/settings.js";


export function getNumClasses() {
    return data.value[0]["ground_truth"].length
}


export function getGlyphSize(xScale, yScale) {
    const desiredGlyphSize = config.models.find(m => m.name === settings.modelName.value)["glyphSize"][settings.dataType.value]
    const glyphBoundingSize = Math.min(
        (xScale(desiredGlyphSize) - xScale(0)),
        (yScale(desiredGlyphSize) - yScale(0))
    )
    const glyphSize = glyphBoundingSize * glyphSizeMultiplier
    return {glyphSize, glyphBoundingSize}
}


export function getStrokeWidth() {
    const desiredGlyphSize = config.models.find(m => m.name === settings.modelName.value)["glyphSize"][settings.dataType.value]
    return desiredGlyphSize * 10
}