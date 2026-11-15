import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def generate_perfect_pipeline_figure(output_path="pipeline.png"):
    # Wide 2-column aspect ratio matching ACM/IEEE template figure*
    fig, ax = plt.subplots(figsize=(16, 5.8), dpi=300)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 5.8)
    ax.axis('off')

    # Color palette (Clean white publication aesthetic with subtle slate accents)
    card_bg = "#FFFFFF"
    card_edge = "#0F172A"       # Crisp dark slate border
    sub_bg = "#F8FAFC"          # Very light subtle gray for subtitle header bar
    text_title = "#0F172A"
    text_sub = "#334155"
    text_bullet = "#1E293B"

    # Card dimensions
    card_w = 3.55
    card_h = 2.45
    
    # 4 columns: x coordinates
    col_x = [0.4, 4.25, 8.1, 11.95]
    row_y = [3.05, 0.35]  # Top row y, Bottom row y

    # Define the 8 Cards in true chronological order
    cards = [
        # --- Top Row (Left to Right: 1 -> 2 -> 3 -> 4) ---
        {
            "id": 1, "col": 0, "row": 0,
            "title": "1. Full-Disk Solar Image",
            "sub": "2048 × 2048 GONG H-α (656.28 nm)",
            "bullets": [
                "Full solar disk input",
                "Extreme scale disparity",
                "Fine spines (< 15 px)"
            ],
            "type": "disk_raw"
        },
        {
            "id": 2, "col": 1, "row": 0,
            "title": "2. Optical Limb Flattening",
            "sub": "Analytical Eddington-Barbier Background",
            "bullets": [
                "Radial median profile",
                "Chromospheric contrast",
                "2-Ch: [I_flat, I_norm]"
            ],
            "type": "disk_flat"
        },
        {
            "id": 3, "col": 2, "row": 0,
            "title": "3. Stage 1: Macro Scout",
            "sub": "YOLO11m @ 1280 × 1280 px",
            "bullets": [
                "STAPLE consensus boxes",
                "High sensitivity recall",
                "20% context expansion"
            ],
            "type": "disk_boxes"
        },
        {
            "id": 4, "col": 3, "row": 0,
            "title": "4. Native Square Crops",
            "sub": "512 × 512 Native Sensor Resolution",
            "bullets": [
                "Extracted at native res",
                "Zero interpolation blur",
                "Captures fine barbs"
            ],
            "type": "crops"
        },

        # --- Bottom Row (Right to Left: 5 -> 6 -> 7 -> 8) ---
        {
            "id": 5, "col": 3, "row": 1,
            "title": "5. Dual U-Net Ensemble",
            "sub": "EfficientNet-B4 + ResNet-34",
            "bullets": [
                "Centerline Dice (clDice)",
                "Soft Dice + BCE loss",
                "Dihedral 4-Fold TTA"
            ],
            "type": "unet"
        },
        {
            "id": 6, "col": 2, "row": 1,
            "title": "6. Bernasconi Reconnection",
            "sub": "Polarity Inversion Line (PIL) Continuity",
            "bullets": [
                "Gap ≤ 35 px, θ ≤ 30°",
                "Bridges broken flux ropes",
                "Respects magnetic PIL"
            ],
            "type": "reconnect"
        },
        {
            "id": 7, "col": 1, "row": 1,
            "title": "7. Absorption Fringe Snapping",
            "sub": "Granulation Gate & Sub-Pixel Shift",
            "bullets": [
                "Trench gate: I_raw ≤ 138",
                "Prunes +7.5% dilation",
                "Sub-pixel shift (-0.23 px)"
            ],
            "type": "snapping"
        },
        {
            "id": 8, "col": 0, "row": 1,
            "title": "8. Collision-Free Allocation",
            "sub": "Non-Overlapping Instance Separation",
            "bullets": [
                "COCO RLE output",
                "4-Fold CV PQ = 0.4706",
                "Public LB = 0.41 (1,199 inst)",
                "Strictly 0 collisions"
            ],
            "type": "allocation"
        }
    ]

    # Draw Cards
    for c in cards:
        cx = col_x[c["col"]]
        cy = row_y[c["row"]]

        # Outer card box
        card_box = patches.FancyBboxPatch(
            (cx, cy), card_w, card_h,
            boxstyle="round,pad=0.08,rounding_size=0.14",
            linewidth=1.4, edgecolor=card_edge, facecolor=card_bg,
            zorder=2
        )
        ax.add_patch(card_box)

        # Title text
        ax.text(cx + card_w/2, cy + card_h - 0.22, c["title"],
                fontsize=9.2, fontweight='bold', color=text_title,
                ha='center', va='center', zorder=3)

        # Subtitle pill / bar
        sub_bar = patches.FancyBboxPatch(
            (cx + 0.12, cy + card_h - 0.58), card_w - 0.24, 0.28,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=0.8, edgecolor="#CBD5E1", facecolor=sub_bg,
            zorder=3
        )
        ax.add_patch(sub_bar)
        ax.text(cx + card_w/2, cy + card_h - 0.44, c["sub"],
                fontsize=6.8, fontweight='semibold', color=text_sub,
                ha='center', va='center', zorder=4)

        # Divider line between thumbnail and bullets
        divider_x = cx + 1.45
        ax.plot([divider_x, divider_x], [cy + 0.2, cy + card_h - 0.72],
                color="#E2E8F0", linewidth=1.2, zorder=3)

        # Render Bullets on the right half
        by = cy + card_h - 0.88
        for b in c["bullets"]:
            ax.plot(divider_x + 0.12, by, marker='o', markersize=3.0, color="#2563EB", zorder=4)
            ax.text(divider_x + 0.22, by, b, fontsize=6.8, color=text_bullet,
                    fontweight='medium', va='center', zorder=4)
            by -= 0.36

        # Render Visual Thumbnail on the left half (cx + 0.15 to cx + 1.35)
        # Visual box coordinates: [cx + 0.18, cy + 0.22, width 1.15, height 1.45]
        vx0, vy0, vw, vh = cx + 0.16, cy + 0.20, 1.18, 1.42
        
        # Draw visual depending on type
        draw_thumbnail(ax, c["type"], vx0, vy0, vw, vh)

    # Draw Inter-card Arrows
    # Row 1 (1 -> 2 -> 3 -> 4)
    arrow_kw = dict(arrowstyle="-|>", mutation_scale=13, linewidth=1.6, color="#0F172A", zorder=6)
    for i in range(3):
        x1 = col_x[i] + card_w + 0.02
        x2 = col_x[i+1] - 0.02
        y = row_y[0] + card_h / 2
        ax.annotate("", xy=(x2, y), xytext=(x1, y), arrowprops=arrow_kw)

    # Turn down from Card 4 to Card 5
    x_turn = col_x[3] + card_w / 2
    y_top = row_y[0] - 0.02
    y_bot = row_y[1] + card_h + 0.02
    ax.annotate("", xy=(x_turn, y_bot), xytext=(x_turn, y_top), arrowprops=arrow_kw)

    # Row 2 (5 -> 6 -> 7 -> 8)
    for i in range(3, 0, -1):
        x1 = col_x[i] - 0.02
        x2 = col_x[i-1] + card_w + 0.02
        y = row_y[1] + card_h / 2
        ax.annotate("", xy=(x2, y), xytext=(x1, y), arrowprops=arrow_kw)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    if output_path != "figures/pipeline.png":
        plt.savefig("figures/pipeline.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Successfully generated perfect pipeline figure at {output_path}")

def draw_thumbnail(ax, vtype, x, y, w, h):
    # Dark thumbnail background for contrast
    bg = patches.Rectangle((x, y), w, h, linewidth=0.8, edgecolor="#CBD5E1", facecolor="#0B0F19", zorder=3)
    ax.add_patch(bg)

    if vtype == "disk_raw":
        # Solar disk circle
        circle = patches.Circle((x + w/2, y + h/2), radius=h*0.42, facecolor="#F8FAFC", edgecolor="#E2E8F0", linewidth=0.8, zorder=4)
        ax.add_patch(circle)
        # Few dark filament lines
        ax.plot([x + w*0.35, x + w*0.55], [y + h*0.62, y + h*0.55], color="#1E293B", linewidth=1.8, zorder=5)
        ax.plot([x + w*0.45, x + w*0.68], [y + h*0.42, y + h*0.38], color="#1E293B", linewidth=1.5, zorder=5)

    elif vtype == "disk_flat":
        # Two small disks: Before -> After
        c1 = patches.Circle((x + w*0.28, y + h/2), radius=h*0.32, facecolor="#CBD5E1", edgecolor="#94A3B8", linewidth=0.6, zorder=4)
        c2 = patches.Circle((x + w*0.72, y + h/2), radius=h*0.32, facecolor="#F1F5F9", edgecolor="#FFFFFF", linewidth=0.6, zorder=4)
        ax.add_patch(c1)
        ax.add_patch(c2)
        ax.annotate("", xy=(x + w*0.56, y + h/2), xytext=(x + w*0.44, y + h/2),
                    arrowprops=dict(arrowstyle="->", color="#38BDF8", linewidth=1.2, mutation_scale=8), zorder=5)

    elif vtype == "disk_boxes":
        circle = patches.Circle((x + w/2, y + h/2), radius=h*0.42, facecolor="#F8FAFC", edgecolor="#E2E8F0", linewidth=0.8, zorder=4)
        ax.add_patch(circle)
        # Yellow detection bounding boxes
        r1 = patches.Rectangle((x + w*0.3, y + h*0.48), 0.38, 0.28, linewidth=1.2, edgecolor="#F59E0B", facecolor="none", zorder=5)
        r2 = patches.Rectangle((x + w*0.52, y + h*0.28), 0.32, 0.24, linewidth=1.2, edgecolor="#F59E0B", facecolor="none", zorder=5)
        ax.add_patch(r1)
        ax.add_patch(r2)

    elif vtype == "crops":
        # 3 stacked square crops
        for i in range(3):
            dx = i * 0.12
            dy = (2 - i) * 0.12
            sq = patches.Rectangle((x + 0.15 + dx, y + 0.25 + dy), 0.65, 0.65,
                                   linewidth=0.8, edgecolor="#94A3B8", facecolor="#1E293B", zorder=4 + i)
            ax.add_patch(sq)
            # Filament curve inside top crop
            if i == 2:
                ax.plot([x + 0.28 + dx, x + 0.58 + dx], [y + 0.4 + dy, y + 0.7 + dy],
                        color="#F8FAFC", linewidth=1.6, zorder=7)

    elif vtype == "unet":
        # Dual U-Net trapezoid symbols
        # EffNet trapezoids
        ax.text(x + w/2, y + h*0.74, "EffNet-B4 U-Net", fontsize=5.8, fontweight='bold', color="#60A5FA", ha='center', zorder=4)
        ax.text(x + w/2, y + h*0.52, "ResNet-34 U-Net", fontsize=5.8, fontweight='bold', color="#C084FC", ha='center', zorder=4)
        # Heatmap probability ribbon
        prob_bar = patches.Rectangle((x + 0.15, y + 0.18), w - 0.3, 0.18,
                                     linewidth=0.6, edgecolor="#38BDF8", facecolor="#0284C7", zorder=4)
        ax.add_patch(prob_bar)
        ax.text(x + w/2, y + 0.27, "Soft Probability Fusion", fontsize=5.0, color="#FFFFFF", ha='center', va='center', zorder=5)

    elif vtype == "reconnect":
        # Step 6: Broken -> Connected
        # Left: broken segments
        ax.plot([x + w*0.15, x + w*0.32], [y + h*0.25, y + h*0.45], color="#FFFFFF", linewidth=2.2, zorder=4)
        ax.plot([x + w*0.38, x + w*0.48], [y + h*0.58, y + h*0.75], color="#FFFFFF", linewidth=2.2, zorder=4)
        # Arrow
        ax.annotate("", xy=(x + w*0.62, y + h*0.5), xytext=(x + w*0.50, y + h*0.5),
                    arrowprops=dict(arrowstyle="->", color="#38BDF8", linewidth=1.4, mutation_scale=8), zorder=5)
        # Right: continuous bridged line
        ax.plot([x + w*0.66, x + w*0.92], [y + h*0.25, y + h*0.75], color="#FFFFFF", linewidth=2.2, zorder=4)
        ax.text(x + w/2, y + 0.12, "PIL Reconnection", fontsize=5.0, color="#94A3B8", ha='center', zorder=4)

    elif vtype == "snapping":
        # Step 7: Thick Bloated -> Snapped Crisp
        # Left: Thick bloated mask
        ax.plot([x + w*0.18, x + w*0.42], [y + h*0.3, y + h*0.75], color="#EF4444", linewidth=5.5, alpha=0.8, zorder=4)
        # Arrow
        ax.annotate("", xy=(x + w*0.60, y + h*0.5), xytext=(x + w*0.48, y + h*0.5),
                    arrowprops=dict(arrowstyle="->", color="#38BDF8", linewidth=1.4, mutation_scale=8), zorder=5)
        # Right: Snapped clean dark core
        ax.plot([x + w*0.66, x + w*0.90], [y + h*0.3, y + h*0.75], color="#22C55E", linewidth=2.0, zorder=4)
        ax.text(x + w/2, y + 0.12, "I_raw <= 138 Gate", fontsize=5.0, color="#94A3B8", ha='center', zorder=4)

    elif vtype == "allocation":
        # Step 8: Multi-color disjoint non-overlapping filaments
        # Red filament
        ax.plot([x + w*0.15, x + w*0.65], [y + h*0.35, y + h*0.82], color="#EF4444", linewidth=2.5, zorder=4)
        # Green filament (separate)
        ax.plot([x + w*0.45, x + w*0.65], [y + h*0.18, y + h*0.25], color="#22C55E", linewidth=2.2, zorder=4)
        # Blue filament (separate)
        ax.plot([x + w*0.75, x + w*0.90], [y + h*0.22, y + h*0.45], color="#3B82F6", linewidth=2.2, zorder=4)
        ax.text(x + w/2, y + 0.12, "0 Pixel Collisions", fontsize=5.0, color="#E2E8F0", fontweight='bold', ha='center', zorder=4)

if __name__ == "__main__":
    generate_perfect_pipeline_figure("pipeline.png")
