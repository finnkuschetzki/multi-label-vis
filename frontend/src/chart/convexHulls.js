import { convexHulls } from "@/stores/data.js"
import { margin, tableau20 } from "@/chart/settings.js"
import { getGlyphSize } from "@/chart/units.js";


export function clearConvexHulls(contentGroup) {
    contentGroup.selectAll(".convex-hulls").remove()
}


export function drawConvexHull(contentGroup, xScale, yScale, featureColumn, label, classIndex) {
    const { glyphSize, glyphBoundingSize } = getGlyphSize(xScale, yScale)

    const points = convexHulls.value[featureColumn][label][classIndex].map(p => [
        xScale(p[0]) + margin.left + glyphBoundingSize/2,
        yScale(p[1]) + margin.bottom + glyphBoundingSize/2
    ])

    contentGroup.append("path")
        .attr("class", "convex-hulls")
        .attr("d", () => {
            let d = ``

            d += `M${points[0][0]},${points[0][1]}`

            for (let i = 1; i < points.length; i++) {
                d += `L${points[i][0]},${points[i][1]}`
            }

            d += `Z`

            return d
        })
        .attr("stroke", tableau20[classIndex])
        .attr("stroke-width", 1)
        .attr("fill", tableau20[classIndex])
        .attr("fill-opacity", 0.25)
}
