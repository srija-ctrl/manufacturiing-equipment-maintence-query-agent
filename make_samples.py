from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

pages = [
    ["HX-400 Hydraulic Press - Maintenance Manual", "",
     "Section 2: Safety. Before any service, lock out and tag out the main disconnect.",
     "Release stored hydraulic pressure by opening bleed valve BV-3 until gauge G1 reads 0 bar.",
     "WARNING: Stored pressure can cause serious injury. Wear eye protection."],
    ["Section 5: Troubleshooting", "",
     "Fault code E-217: Low hydraulic pressure.",
     "1. Check reservoir oil level; refill with ISO VG 46 oil to the MAX mark.",
     "2. Inspect suction filter F2 for clogging; replace if differential pressure exceeds 1.5 bar.",
     "3. Verify relief valve RV-1 is set to 210 bar.",
     "Torque for pump mounting bolts: refer to the OEM appendix (not included)."],
    ["Section 7: Preventive maintenance schedule", "",
     "Daily: check oil level and inspect hoses for leaks.",
     "Every 500 hours: replace suction filter F2.",
     "Every 2000 hours: replace hydraulic oil and clean the reservoir."],
]
c = canvas.Canvas("tests/HX-400_manual.pdf", pagesize=letter)
for lines in pages:
    y = 740
    for line in lines:
        c.drawString(72, y, line); y -= 18
    c.showPage()
c.save()

open("tests/press_log.txt", "w").write(
    "2026-03-02 HX-400 E-217 low pressure. Replaced suction filter F2. Pressure restored.\n"
    "2026-04-19 HX-400 E-217 again. Found loose hose clamp at pump inlet; tightened.\n"
)
open("tests/empty.txt", "w").write("   \n")
