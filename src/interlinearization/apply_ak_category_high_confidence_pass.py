#!/usr/bin/env python3
"""Upsert approved high-confidence A.K. category-text corrections."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from auto_interlinearize import ROOT, normalize_key


CORRECTIONS_PATH = ROOT / "interlinearization/review_corrections.tsv"
OUTPUT_DIR = ROOT / "interlinearization/outputs"
AUDIT_PATH = ROOT / "interlinearization/outputs/ak_category_high_confidence_pass_20260430.tsv"
SOURCE = "ak_category_high_confidence_pass_20260430"

FIELDS = [
    "form",
    "corrected_form",
    "lemma",
    "gloss_en",
    "gloss_tr",
    "pos",
    "confidence",
    "source",
    "note",
    "status",
    "lesson_ids",
    "line_ids",
    "updated_at",
    "morphemes",
]


def morph(*items: tuple[str, str, str]) -> str:
    return json.dumps([{"form": form, "gloss": gloss, "pos": pos} for form, gloss, pos in items], ensure_ascii=False)


RAW_CORRECTIONS = [
    # Teaching-frame demonstratives and question words.
    ("ow", "ew", "that/it", "DEM", 0.97, "", "A.K. category introductions repeatedly use Ow/Owı where Turkish translations have o/bu; standard cognate is ew."),
    ("owı", "ew", "that/it", "DEM", 0.95, "", "A.K. variant in introductions such as Owı qwımaş; translated as o/bu."),
    ("êyi", "eyi", "they/those", "DEM", 0.96, "", "Harun has eyi as plural demonstrative/pronoun; A.K. translations use onlar/bunlar."),
    ("eyi", "eyi", "they/those", "DEM", 0.94, "", "Unraised spelling variant of A.K. Êyi."),
    ("çada", "ça", "where-LOC", "ADV", 0.96, morph(("ça", "where", "ADV"), ("da", "LOC", "CASE")), "Harun gives ça as where; A.K. question lines translate ha çada as nerede."),
    ("cada", "ça", "where-LOC", "ADV", 0.92, morph(("ca", "where", "ADV"), ("da", "LOC", "CASE")), "Unaccented/dialect spelling of çada in where-question frames."),
    ("ça", "ça", "where", "ADV", 0.9, "", "Harun gives ça as where; in A.K. split spelling ça da means where-at."),
    ("kûmcada", "kûmca", "where-LOC", "ADV", 0.88, morph(("kûm", "which", "PRON"), ("ca", "place", "N"), ("da", "LOC", "CASE")), "Recurring A.K. question word in nerede/hangi-place frames; morpheme split is provisional but meaning is clear."),
    ("êr", "êr", "today", "ADV", 0.95, "", "Calendar and future-weather categories translate Êr as bugün."),

    # Common category nouns/adjectives with strong dictionary or category evidence.
    ("hûlik", "hûlik", "baby", "N", 0.96, "", "Baby-care categories repeatedly use hûlik for Turkish bebek."),
    ("hûlık", "hûlik", "baby", "N", 0.93, "", "Dotless spelling variant of hûlik baby in baby-care categories."),
    ("mwêrik", "mwêrik", "man", "N", 0.96, "", "Body/emotion categories use mwêrik/mwerik where translations have adam/man."),
    ("mwerik", "mwêrik", "man", "N", 0.96, "", "Unraised spelling variant of mwêrik man; overrides false murık spinal-bone match."),
    ("mwerık", "mwêrik", "man", "N", 0.94, "", "Dotless spelling variant of mwêrik man; overrides false murık spinal-bone match."),
    ("râqûm", "râqûm", "number", "N", 0.95, "", "Number category frames use râqûm before numerals; Turkish title/category is numbers."),
    ("gıcik", "gıcık", "hair", "N", 0.96, "", "Harun has gıcık/gıjık hair; A.K. hair/baby-care categories confirm."),
    ("gıcık", "gıcık", "hair", "N", 0.96, "", "Harun has gıcık/gıjık hair; A.K. hair/baby-care categories confirm."),
    ("argwêş", "argwêş", "rabbit", "N", 0.95, "", "Benzerlik category translations use tavşan/rabbit for argwêş."),
    ("argweş", "argwêş", "rabbit", "N", 0.93, "", "Unraised spelling variant of argwêş rabbit."),
    ("çerdek", "çerdek", "seed/kernel", "N", 0.94, "", "Nut/dry-food categories use çerdek for çekirdek seed/kernel."),
    ("mirçik", "mirçik", "bird", "N", 0.95, "", "Animal and verb categories translate mirçik as kuş/bird; also used for sparrow in bird category."),
    ("miçik", "mirçik", "bird", "N", 0.9, "", "Reduced/variant bird form in bird category."),
    ("owk", "awk", "water", "N", 0.95, "", "Harun has awk water; A.K. water/garden categories use owk/howk for water."),
    ("awk", "awk", "water", "N", 0.96, "", "Harun has awk water."),
    ("howk", "awk", "water", "N", 0.92, "", "A.K. h-prothetic/rounded variant of awk water in water-related contexts."),
    ("sya", "siya", "black", "ADJ", 0.95, "", "DEZD and Harun have siya black; A.K. color category uses sya/siya."),
    ("siya", "siya", "black", "ADJ", 0.95, "", "DEZD and Harun have siya black."),
    ("swır", "swır", "red", "ADJ", 0.94, "", "Color and hair categories translate swır as kırmızı/red; overrides false sor salty match."),
    ("şwış", "şwış", "glass/cup", "N", 0.94, "", "Drink/kitchen categories use şwış for glass/cup; prevents false sauce match."),
    ("şwışê", "şwış", "glass/cup-EZ", "N", 0.93, morph(("şwış", "glass/cup", "N"), ("ê", "EZ", "EZ")), "Drink/kitchen categories use şwışê for tea glass, coffee-side water glass, and similar containers; overrides false sauce match."),
    ("şwışi", "şwış", "glass/cup-OBL", "N", 0.9, morph(("şwış", "glass/cup", "N"), ("i", "OBL", "CASE")), "Oblique/raised variant of şwış glass/cup in kitchen contexts."),
    ("fistix", "fistix", "nut/pistachio", "N", 0.9, "", "Snack category uses fistix in Antep fıstığı, kaju fıstığı, and yer fıstığı frames."),
    ("heş", "heş", "bear", "N", 0.96, "", "DEZD and Harun have heş bear; A.K. animal/toy-bear category confirms."),
    ("myeşna", "myeşna", "sheep", "N", 0.96, "", "Animal and similarity categories translate myeşna as koyun/sheep."),
    ("miyeşna", "myeşna", "sheep", "N", 0.94, "", "Expanded spelling variant of myeşna sheep."),
    ("hor", "hor", "cloud", "N", 0.95, "", "Benzerlik category translations use bulut/cloud for hor."),
    ("kılse", "kılse", "chair", "N", 0.92, "", "House/location categories translate kılse as sandalye/chair."),
    ("hâfte", "hâfte", "week", "N", 0.95, "", "Calendar categories translate hâfte as week."),
    ("tabaxê", "tabax", "plate-EZ", "N", 0.92, morph(("tabax", "plate", "N"), ("ê", "EZ", "EZ")), "Food categories use tabaxê in plate/dish expressions."),
    ("cınawır", "cınawır", "monster/beast", "N", 0.9, "", "Animal/legend categories use cınawır; cognate and category context indicate monster/beast."),
    ("kweçik", "kweçik", "spoon", "N", 0.95, "", "Kitchen categories translate kweçik/kweçika as kaşık/spoon."),
    ("kweçika", "kweçik", "spoon-F.SG", "N", 0.95, morph(("kweçik", "spoon", "N"), ("a", "F.SG", "AGR")), "Kitchen categories translate kweçika as kaşık/spoon with feminine ending."),
    ("kweçikê", "kweçik", "spoon-EZ", "N", 0.93, morph(("kweçik", "spoon", "N"), ("ê", "EZ", "EZ")), "Kitchen categories use kweçikê in spoon compounds."),
    ("vêw", "vêw", "bride/daughter-in-law", "N", 0.9, "", "Harun has vêw/veyv gelin; family categories use vêw/vêwê ma."),
    ("vêwê", "vêw", "bride/daughter-in-law-EZ", "N", 0.9, morph(("vêw", "bride/daughter-in-law", "N"), ("ê", "EZ", "EZ")), "Family categories use vêwê ma for in-law/bride terms; weather phrase remains phrase-level review if needed."),
    ("qıc", "qıc", "child", "N", 0.93, "", "Harun gives qıc small/small child; A.K. child/baby contexts favor child."),
    ("qic", "qıc", "child", "N", 0.92, "", "Unaccented spelling of qıc child in A.K. child/baby contexts."),
    ("qıcû", "qıc", "child-PL.OBL", "N", 0.91, morph(("qıc", "child", "N"), ("û", "PL.OBL", "NUM")), "Reduced oblique plural form in baby-item compounds."),
    ("qıcik", "qıcik", "small/little", "ADJ", 0.92, "", "A.K. uses qıcik in small bowl and baby/child expressions."),
    ("faslyê", "faslye", "bean", "N", 0.93, "", "Food categories use Faslyê for fasulye/beans."),
    ("gırıgıl", "gırıgıl", "kernel/grain", "N", 0.92, "", "Food categories use gırıgıl for mısır tanesi and popcorn expressions."),
    ("gırıgil", "gırıgıl", "kernel/grain", "N", 0.9, "", "Unraised spelling variant of gırıgıl kernel/grain."),
    ("qâ'hwi", "qâ'hwe", "coffee", "N", 0.93, "", "Kitchen/drink categories use qâ'hwi/qâ'hwe in coffee contexts."),
    ("qâ'hwe", "qâ'hwe", "coffee", "N", 0.93, "", "Drink category coffee form."),
    ("qâlêm", "qâlêm", "pen/pencil", "N", 0.92, "", "Stationery category uses qâlêm in pencil/colored-pencil compounds."),
    ("tâxtê", "tâxt", "board-EZ", "N", 0.9, morph(("tâxt", "board", "N"), ("ê", "EZ", "EZ")), "Kitchen category uses tâxtê in cutting-board expressions."),
    ("tpap", "tpap", "ball", "N", 0.94, "", "Game/child action categories use tpap for top/ball; overrides false tuap cannon match."),
    ("tpapê", "tpap", "ball-EZ", "N", 0.9, morph(("tpap", "ball", "N"), ("ê", "EZ", "EZ")), "Ezafe/oblique form of tpap ball."),

    # Dictionary matches that were correct but below the auto-review threshold.
    ("darê", "dar", "tree-EZ", "N", 0.94, morph(("dar", "tree", "N"), ("ê", "EZ", "EZ")), "Tree category confirms darê in tree compounds."),
    ("çay", "çay", "tea", "N", 0.95, "", "Drink category uses çay in tea compounds."),
    ("çayê", "çay", "tea-EZ", "N", 0.94, morph(("çay", "tea", "N"), ("ê", "EZ", "EZ")), "Drink category uses çayê in tea compounds; overrides false place match."),
    ("gışt", "gışt", "finger", "N", 0.95, "", "Harun/Kaikki and body-part categories confirm gışt finger."),
    ("gışta", "gışt", "finger-F.SG", "N", 0.93, morph(("gışt", "finger", "N"), ("a", "F.SG", "AGR")), "Body-part categories use gışta for finger/thumb compounds."),
    ("gıştê", "gışt", "finger-EZ", "N", 0.91, morph(("gışt", "finger", "N"), ("ê", "EZ", "EZ")), "Body-part category ezafe form of gışt finger."),
    ("kerra", "kerra", "stone/rock", "N", 0.94, "", "Harun dictionary plus stone/rock category context."),
    ("kerrawo", "kerra", "stone/rock-and", "N", 0.92, morph(("kerra", "stone/rock", "N"), ("wo", "and", "CONJ")), "Bağlaç category uses kerrawo as kerra plus cliticized and."),
    ("mas", "mas", "fish", "N", 0.94, "", "Harun dictionary and animal categories confirm fish."),
    ("masê", "mas", "fish-EZ", "N", 0.92, morph(("mas", "fish", "N"), ("ê", "EZ", "EZ")), "Animal categories use masê in fish compounds."),
    ("bıra", "bıra", "brother", "N", 0.94, "", "Family/action categories use bıra/bırê for brother."),
    ("bırê", "bıra", "brother-OBL/EZ", "N", 0.92, morph(("bır", "brother", "N"), ("ê", "OBL/EZ", "CASE")), "Action categories translate bırê xwe as his brother; overrides false well match."),
    ("şit", "şıt", "milk", "N", 0.94, "", "Harun dictionary plus drink/baby-care categories confirm milk."),
    ("pyûnz", "pyûnz", "onion", "N", 0.94, "", "Harun dictionary and vegetable category confirm onion."),
    ("kûma", "kûm", "which/who-F.SG", "PRON", 0.91, morph(("kûm", "which/who", "PRON"), ("a", "F.SG", "AGR")), "Question translations with hangi confirm kûma as feminine/agreement form of kûm."),
    ("daro", "dar", "tree-and", "N", 0.9, morph(("dar", "tree", "N"), ("o", "and", "CONJ")), "Bağlaç category uses daro as tree plus cliticized conjunction before the next item."),
    ("kıncû", "kınc", "clothes/laundry-PL.OBL", "N", 0.91, morph(("kınc", "clothes/laundry", "N"), ("û", "PL.OBL", "NUM")), "Harun gives kınc clothes/laundry; baby-care and house categories confirm."),
    ("mına", "mı", "1SG.POSS", "PRON", 0.93, "", "Family-category possessive phrases translate mına as my, not cover-F.SG."),
    ("darû", "dar", "tree-PL.OBL", "N", 0.9, morph(("dar", "tree", "N"), ("û", "PL.OBL", "NUM")), "Garden-action category uses darû for trees in grafting/pruning contexts; overrides medicine match there."),
    ("şwena", "şwenayış", "wash-IPFV-F.SG", "V", 0.92, morph(("şw", "wash", "V"), ("en", "IPFV", "ASP"), ("a", "F.SG", "AGR")), "Baby-care and present-tense categories translate şwena as washing."),
    ("vecena", "vecnayış", "take.out-IPFV-F.SG", "V", 0.9, morph(("vec", "take.out", "V"), ("en", "IPFV", "ASP"), ("a", "F.SG", "AGR")), "Baby-care gas-removal contexts translate vecena as take out/bring out."),
    ("ârâbe", "ârâbe", "car", "N", 0.93, "", "Children/action categories use ârâbe for car."),
    ("hawa", "hawa", "air/weather", "N", 0.91, "", "Flight/weather contexts use hawa as air/weather; overrides bark match."),
    ("dıwar", "dıwar", "livestock", "N", 0.88, "", "Animal category uses dıwar for livestock classes; keep as category-backed rather than wall."),
    ("wıskrawa", "wıskra", "bowl-F.SG", "N", 0.9, morph(("wıskra", "bowl", "N"), ("a", "F.SG", "AGR")), "Kitchen/bowl adjective categories use wıskrawa for bowl."),
    ("sandalyê", "sandalye", "chair-EZ", "N", 0.9, morph(("sandalye", "chair", "N"), ("ê", "EZ", "EZ")), "Baby-item category uses sandalyê in high-chair compound."),
    ("dınûn", "dınûn", "teeth", "N", 0.93, "", "Body and tooth-brush categories use dınûn for teeth; overrides false religion match."),
    ("dınûnû", "dınûn", "teeth-PL.OBL", "N", 0.91, morph(("dınûn", "teeth", "N"), ("û", "PL.OBL", "NUM")), "Tooth-brush and baby-care categories use dınûnû for teeth in oblique compounds."),
    ("nimnen", "nimnayış", "hide-IPFV", "V", 0.9, morph(("nimn", "hide", "V"), ("en", "IPFV", "ASP")), "Two-brothers category translates xwe nimnen as hiding oneself."),
    ("hârzen", "hârzayış", "throw-IPFV", "V", 0.9, morph(("hârz", "throw", "V"), ("en", "IPFV", "ASP")), "Two-brothers ball category uses hârzen for throwing."),
    ("giryena", "giryanayış", "boil-IPFV-F.SG", "V", 0.9, morph(("giry", "boil", "V"), ("en", "IPFV", "ASP"), ("a", "F.SG", "AGR")), "Water category translates awk ha giryena as water is boiling."),
    ("naylûn", "naylûn", "nylon/plastic", "N", 0.88, "", "Material/container categories use naylûn for nylon/plastic."),
    ("şûne", "şûne", "comb", "N", 0.82, "", "Hair-combing baby-care frames identify şûne as comb; kept below high-confidence threshold because sink-item context should still be reviewed."),

    # Recurrent verbs/light elements where translations make the analysis clear.
    ("mûnen", "mûnayış", "resemble-IPFV", "V", 0.92, morph(("mûn", "resemble", "V"), ("en", "IPFV", "ASP")), "Benzerlik category translations use benzemek/resemble for mûnen."),
    ("mûnena", "mûnayış", "resemble-IPFV-F.SG", "V", 0.92, morph(("mûn", "resemble", "V"), ("en", "IPFV", "ASP"), ("a", "F.SG", "AGR")), "Benzerlik category translations use benzemek/resemble for mûnena."),
    ("mûnên", "mûnayış", "resemble-IPFV.PL", "V", 0.9, morph(("mûn", "resemble", "V"), ("ên", "IPFV.PL", "ASP")), "Benzerlik category plural/raised imperfective form of resemble."),
    ("nımûnên", "mûnayış", "NEG-resemble-IPFV.PL", "V", 0.9, morph(("nı", "NEG", "POL"), ("mûn", "resemble", "V"), ("ên", "IPFV.PL", "ASP")), "Benzerlik category translations use benzemiyor/not resemble."),
    ("hesterena", "hesterayış", "wipe-IPFV-F.SG", "V", 0.91, morph(("hester", "wipe", "V"), ("en", "IPFV", "ASP"), ("a", "F.SG", "AGR")), "Baby-care and cleaning categories translate hesterena as silmek/wipe."),
    ("cir", "cir", "calm", "NVE", 0.86, "", "Baby-care expression kena cir means soothe/calm; keep as nonverbal element for review if needed."),

    # Second category-text pass: repeated forms whose category translations and dictionaries agree.
    ("binû", "bin", "other-PL.OBL", "PRON", 0.91, morph(("bin", "other", "PRON"), ("û", "PL.OBL", "NUM")), "Similarity rows use yo binû for one another/each other; suffix analysis is consistent with oblique plural -û."),
    ("têrlik", "terlik", "slippers", "N", 0.94, "", "Clothing/accessory rows translate Têrlik as terlik/slippers."),
    ("pârê", "pâre", "money/currency", "N", 0.92, "", "Shopping/money rows use Pârê in dollar, euro, coin, and Turkish-lira expressions; this is the money sense, not piece/part."),
    ("renga", "reng", "color-F.SG", "N", 0.96, morph(("reng", "color", "N"), ("a", "F.SG", "AGR")), "Color category uses Renga before color adjectives such as swır and sya; Harun/DEZD support reng as color."),
    ("mişmiş", "mışmış", "apricot", "N", 0.94, "", "Fruit and dried-fruit rows translate Mişmiş as apricot/kayısı."),
    ("vistore", "vıstore", "in-law", "N", 0.92, "", "Family in-law rows use Vistore for kayınpeder/kayınbirader-type in-law meanings; keep the gloss broad because the exact kin role varies by row."),
    ("çaydûn", "çaydûn", "teapot", "N", 0.95, "", "Kitchen rows translate Çaydûn as çaydanlık/teapot; dictionary resources support çaydun."),
    ("cılû", "cıl", "clothes/bedding-PL.OBL", "N", 0.9, morph(("cıl", "clothes/bedding", "N"), ("û", "PL.OBL", "NUM")), "Household rows use cılû in clothing/bedding contexts; Harun gives cıl as clothing/bedding."),
    ("gâz", "gâz", "gas", "N", 0.93, "", "Baby-care rows translate Bebeğin gazını çıkarıyor; this is gas, not the bite verb."),
    ("lawnena", "lawnenayış", "breastfeed-IPFV-F.SG", "V", 0.92, morph(("lawn", "breastfeed", "V"), ("en", "IPFV", "ASP"), ("a", "F.SG", "AGR")), "Baby-care rows translate hûlik lawnena as breastfeeding/nursing the baby."),
    ("masi", "mas", "fish-OBL/EZ", "N", 0.91, morph(("mas", "fish", "N"), ("i", "OBL/EZ", "CASE")), "Animal body-part and young-animal rows use masi in fish compounds."),
    ("masû", "mas", "fish-PL.OBL", "N", 0.89, morph(("mas", "fish", "N"), ("û", "PL.OBL", "NUM")), "Animal/rural rows use masû in fish/eel expressions; supported by the accepted mas fish analysis."),
    ("mirçikê", "mirçik", "bird-EZ/OBL", "N", 0.92, morph(("mirçik", "bird", "N"), ("ê", "EZ/OBL", "CASE")), "Bird/body-part rows use Mirçikê in bird compounds."),
    ("miçikê", "mirçik", "bird-EZ/OBL", "N", 0.9, morph(("miçik", "bird", "N"), ("ê", "EZ/OBL", "CASE")), "Reduced bird form in the bird category; matches accepted mirçik/miçik bird analyses."),
    ("rûn", "rûn", "oil/fat", "N", 0.94, "", "Staple-food rows translate Rûn as yağ/tereyağı; overrides the unrelated sweep sense."),
    ("nyûn", "nyûn", "between/among", "ADP", 0.92, "", "Location rows use nyûn for between/among, as in between two chairs; this is distinct from nyû nyê 'or not'."),
    ("kılsida", "kılse", "chair-LOC", "N", 0.9, morph(("kılse", "chair", "N"), ("da", "LOC", "CASE")), "Location rows use kılsida in chair-location frames; analyze as chair plus locative -da."),
    ("fırfırik", "fırfırik", "fan", "N", 0.9, "", "Household/lavabo rows use Fırfırik for fan/ventilator, extending the dictionary wind-turning toy sense."),
    ("ışlig", "ışlig", "shirt", "N", 0.94, "", "Clothing rows translate Işlig as shirt/gömlek."),
    ("gwel", "gwel", "lake", "N", 0.92, "", "Nature/location rows translate Gwel as lake/göl."),
    ("kerdiş", "kerdış", "do/make.NMLZ", "VN", 0.88, "", "Game/activity and stationery compounds use kerdiş as a doing/making nominalization."),
    ("çûntê", "çûnte", "bag-EZ", "N", 0.92, morph(("çûnte", "bag", "N"), ("ê", "EZ", "EZ")), "Household/shopping rows use Çûntê in bag compounds."),
    ("resimda", "resım", "picture-LOC", "N", 0.94, morph(("resım", "picture", "N"), ("da", "LOC", "CASE")), "Location/water rows use resimda where the Turkish metadata has resim; analyze as picture plus locative -da."),
    ("parçê", "parçe", "piece-EZ", "N", 0.91, morph(("parçe", "piece", "N"), ("ê", "EZ", "EZ")), "Puzzle/block rows use Parçê in piece/part compounds."),
    ("mwêriko", "mwêrik", "man-M.SG.COP", "N", 0.9, morph(("mwêrik", "man", "N"), ("o", "M.SG.COP", "COP")), "Nominal man rows use mwêriko as man plus masculine/copular -o; this is not the verbal A.K. 1SG -o."),
    ("mêywê", "mêywe", "fruit", "N", 0.92, "", "Food rows use Mêywê for fruit/meyve."),
    ("incir", "incir", "fig", "N", 0.92, "", "Fruit rows translate incir as fig."),
    ("hortûm", "hortûm", "hose", "N", 0.92, "", "Tool/garden rows translate Hortûm as hose/hortum."),
    ("rûwo", "rûw", "river-and", "N", 0.9, morph(("rûw", "river", "N"), ("o", "and", "CONJ")), "Conjunction rows use rûwo as river plus cliticized and."),
    ("rûwê", "rûw", "river-EZ/OBL", "N", 0.9, morph(("rûw", "river", "N"), ("ê", "EZ/OBL", "CASE")), "Conjunction/location rows use rûwê in river expressions."),
    ("mirçiko", "mirçik", "bird-and", "N", 0.9, morph(("mirçik", "bird", "N"), ("o", "and", "CONJ")), "Conjunction rows use mirçiko as bird plus cliticized and."),
    ("çiçeg", "çiçeg", "flower", "N", 0.91, "", "Conjunction rows translate çiçeg as flower/çiçek."),
    ("swal", "swal", "salt", "N", 0.9, "", "Cooking rows use swal for salt, especially Ha swal kena de 'adds salt'; overrides the false shoe match."),
    ("batûnyê", "batûnye", "blanket-EZ", "N", 0.9, morph(("batûnye", "blanket", "N"), ("ê", "EZ", "EZ")), "Baby-item rows translate Batûnyê as blanket/battaniye."),
    ("âmzik", "âmzik", "pacifier", "N", 0.9, "", "Baby-item rows translate Âmzik as pacifier/emzik."),
    ("hılikû", "hûlik", "baby-PL.OBL", "N", 0.9, morph(("hûlik", "baby", "N"), ("û", "PL.OBL", "NUM")), "Baby-item compounds such as mamê hılikû and zıbınê hılikû use the oblique plural baby form."),
    ("têyinç", "têyinç", "wild grape", "N", 0.9, "", "Plant rows translate Têyinç as wild grape/yabani üzüm."),
    ("tırkwaz", "tırkwaz", "turquoise", "N", 0.9, "", "Gem/color rows translate Tırkwaz as turquoise."),
    ("qwımaş", "qwımaş", "fabric", "N", 0.92, "", "Sewing/material rows translate Qwımaş as fabric/kumaş."),
    ("asmin", "asmin", "sky", "N", 0.9, "", "Nature rows translate Asmin as sky/gökyüzü."),
    ("sarê", "sar", "head-EZ/OBL", "N", 0.92, morph(("sar", "head", "N"), ("ê", "EZ/OBL", "CASE")), "Body and bed/headboard rows use sarê as head in compounds; overrides false people match."),
    ("kılso", "kılse", "chair-M.SG.COP", "N", 0.9, morph(("kılse", "chair", "N"), ("o", "M.SG.COP", "COP")), "House/rural rows use kılso in chair introductions."),
    ("telefûn", "telefûn", "telephone", "N", 0.93, "", "Household rows translate Telefûn as telephone/telefon; overrides the false telef verb match."),
    ("sawın", "sawın", "soap", "N", 0.9, "", "Household/lavabo rows use Sawın for soap/sabun."),
    ("fırçê", "fırçe", "brush-EZ", "N", 0.9, morph(("fırçe", "brush", "N"), ("ê", "EZ", "EZ")), "Lavabo rows use Fırçê dınûnû for toothbrush, so fırçê is brush with ezafe."),
    ("cıqli", "cıqli", "tickle/tickling", "NVE", 0.9, "", "Baby-care rows translate cıqli kena as tickling the baby."),
    ("hâsrû", "hâsrû", "tears", "N", 0.88, "", "Baby-care rows translate hâsrû in wiping the baby's tears."),
    ("hirı", "hirı", "three", "NUM", 0.94, "", "Calendar rows translate Hirı rwec as three days."),
    ("rwec", "rwêc", "day", "N", 0.9, "", "Calendar rows use rwec/rwêc for day."),
    ("rwecê", "rwêc", "day-EZ/OBL", "N", 0.9, morph(("rwêc", "day", "N"), ("ê", "EZ/OBL", "CASE")), "Calendar rows use rwecê in day expressions."),
    ("hêdi", "hêdi", "slowly/softly", "ADV", 0.9, "", "Adverb rows translate Hêdi as yavaş and use it for both slow eating and speaking softly/quietly."),
    ("zûma", "zûma", "bridegroom/son-in-law", "N", 0.91, "", "Family rows translate Zûma as damat and use it for an in-law spouse."),
    ("aşçi", "aşçi", "cook", "N", 0.92, "", "Profession rows translate Aşçi as cook/aşçı."),
    ("balûn", "balûn", "balloon", "N", 0.92, "", "Game/activity rows translate Balûn as balloon/balon."),
    ("balcûn", "balcûn", "eggplant", "N", 0.92, "", "Vegetable rows translate Balcûn as eggplant/patlıcan."),
    ("bamyê", "bamye", "okra", "N", 0.92, "", "Vegetable rows translate Bamyê as okra/bamya."),
    ("babi", "bab", "father-OBL/EZ", "N", 0.9, morph(("bab", "father", "N"), ("i", "OBL/EZ", "CASE")), "Family rows use babi in father-side and from-father kinship expressions."),
    ("pisinga", "pising", "cat-F.SG", "N", 0.95, morph(("pising", "cat", "N"), ("a", "F.SG", "AGR")), "Animal rows use pisinga for cat and in big-cat compounds; category metadata confirms the cat sense."),
    ("qıcûno", "qıc", "child-PL.OBL-M.SG.COP", "N", 0.88, morph(("qıc", "child", "N"), ("û", "PL.OBL", "NUM"), ("o", "M.SG.COP", "COP")), "Baby-item introduction rows use qıcûno in child/baby-object compounds."),
    ("bêrzin", "bêrzin", "pillow/cushion", "N", 0.91, "", "Household and bedroom rows translate Bêrzin as pillow/cushion/yastık/kırlent; overrides the false subjunctive throw parse."),
    ("wırdayiş", "wırdayış", "cutting.NMLZ", "VN", 0.9, "", "Kitchen rows use wırdayiş in cutting-board compounds."),
    ("çarnena", "çarnayış", "carry.on.back-IPFV-F.SG", "V", 0.9, morph(("çarn", "carry.on.back", "V"), ("en", "IPFV", "ASP"), ("a", "F.SG", "AGR")), "Baby-care rows translate paşda çarnena as carrying the baby on the back."),
    ("rweşena", "rweşayış", "shake.out-IPFV-F.SG", "V", 0.9, morph(("rweş", "shake.out", "V"), ("en", "IPFV", "ASP"), ("a", "F.SG", "AGR")), "Housework rows translate tâberra rweşena as shaking out a tablecloth."),
    ("qwal", "qwal", "arm", "N", 0.9, "", "Body-part rows translate Qwal as arm and use bin qwal for armpit/under-arm."),
    ("gırrda", "gırrd", "big-F.SG", "ADJ", 0.88, morph(("gırrd", "big", "ADJ"), ("a", "F.SG", "AGR")), "Animal rows use pisinga gırrda for big-cat expressions such as leopard/lynx."),
    ("finena", "finayış", "place/hang-IPFV-F.SG", "V", 0.88, morph(("fin", "place/hang", "V"), ("en", "IPFV", "ASP"), ("a", "F.SG", "AGR")), "User correction plus image/Turkish contexts: finena ha is hanging laundry; finena rwı/rwe is putting away or placing items. Phrase rows disambiguate the construction."),
    ("finena ha", "ha finayış", "hang up", "PHRASE", 0.96, "", "User correction: finena ha is hanging up. Category Turkish and linked image are Çamaşır asıyor."),
    ("finena rwı", "rwe finayış", "put away / put in place", "PHRASE", 0.95, "", "User correction: finena rwı/rwe is putting away or putting in its place. Category Turkish and linked image are Konserveleri dolaba yerleştiriyor."),
    ("finena rwe", "rwe finayış", "put away / put in place", "PHRASE", 0.95, "", "Spelling variant of finena rwı for the same light-verb construction."),
    ("tırşik", "tırşik", "sour/cress", "N/ADJ", 0.88, "", "User notes sour; category image/title also show Tere for the herb and Nar ekşisi for the sour pomegranate compound, so the standalone form is kept broad."),
    ("tırşik hânnar", "tırşik hânnar", "pomegranate sour/molasses", "PHRASE", 0.92, "", "Nar ekşisi category and image support this phrase gloss."),
    ("tırşik hannar", "tırşik hânnar", "pomegranate sour/molasses", "PHRASE", 0.88, "", "Unaccented surface variant of the Nar ekşisi phrase."),
    ("hûna", "hûna", "out/asleep", "NVE", 0.83, "", "Phrase-sensitive nonverbal element; kept yellow while phrase rows handle extinguish and putting to sleep."),
    ("ken hûna", "hûna kerdış", "extinguish / put out", "PHRASE", 0.92, "", "Ateşi söndürüyor Turkish context and image support the phrase gloss."),
    ("kena hûna", "hûna kerdış", "put to sleep", "PHRASE", 0.92, "", "Bebeği uyutuyor Turkish context and image support the phrase gloss."),
    ("howli", "hewli", "balcony/terrace", "N", 0.93, "", "Balkon and Köy balkonu category images support balcony/terrace; local Harun evidence has holı/hewli."),
    ("çartele", "çartele", "thistle", "N", 0.91, "", "Devedikeni and Top diken category rows support thistle."),
    ("çartelo", "çartele", "thistle-M.SG.COP", "N", 0.88, morph(("çartele", "thistle", "N"), ("o", "M.SG.COP", "COP")), "Introduction rows use çartelo for thistle."),
    ("qwılbe", "qwılbe", "hoe", "N", 0.92, "", "Çapa category translation and image support hoe."),
    ("tpêrzin", "tpêrzin", "axe", "N", 0.92, "", "Balta and Savaş baltası contexts and images support axe."),
    ("mêsin", "mêsin", "ewer/jug", "N", 0.9, "", "İbrik and Güğüm contexts and images support ewer/jug."),
    ("dâst", "dâst", "hand", "N", 0.94, "", "Body-part rows and image show El."),
    ("dâst hârzân", "dest eştiş", "help/lend a hand", "PHRASE", 0.9, "", "Kardeşinin topu almasına yardım ediyor context and image support a help/lend-a-hand phrase."),
    ("dest hârzân", "dest eştiş", "help/lend a hand", "PHRASE", 0.9, "", "Surface variant in the same helping context."),
    ("şûm", "şûm", "dinner/evening meal", "N", 0.92, "", "Akşam yemeği and cook-dinner contexts support dinner/evening meal."),
    ("sytlacir", "sytlac", "rice.pudding-LOC/onto", "N", 0.88, morph(("sytlac", "rice.pudding", "N"), ("r", "LOC/ONTO", "CASE")), "Sütlacın üzerine ceviz serpiyor context supports rice pudding with an oblique/locative ending."),
    ("sêr sytlacir", "sytlac ser", "on the rice pudding", "PHRASE", 0.9, "", "The Sütlacın üzerine context makes the phrase relation explicit."),
    ("gwazû", "gwaz", "walnut-PL.OBL", "N", 0.9, morph(("gwaz", "walnut", "N"), ("û", "PL.OBL", "NUM")), "The Sütlacın üzerine ceviz serpiyor category uses gwazû for walnuts."),
    ("otobys", "otobys", "bus", "N", 0.92, "", "Otobüs and bus-travel images support bus."),
    ("pirik", "pirik", "grandparent/elder", "N", 0.88, "", "A.K. category rows use pirik across Büyükanne and Dede, so this stays broad rather than gender-specific."),
    ("âm", "âm", "paternal aunt", "N", 0.92, "", "Hala context and local Harun am/aem evidence support paternal aunt."),
    ("mêrdimcêy", "mêrdimcêy", "relative", "N", 0.88, "", "Uzaktan akraba context supports relative."),
    ("dbirr ra mêrdimcêy", "dbirr ra mêrdimcêy", "distant relative", "PHRASE", 0.92, "", "Uzaktan akraba context supports the full phrase."),
    ("bırêcêy", "bırêcêy", "step-sibling", "N", 0.9, "", "Babadan/Anneden üvey kardeş contexts support step-sibling."),
    ("babi ra bırêcêy", "babi ra bırêcêy", "paternal step-sibling", "PHRASE", 0.92, "", "Babadan üvey kardeş context."),
    ("dadi ra bırêcêy", "dadi ra bırêcêy", "maternal step-sibling", "PHRASE", 0.9, "", "Anneden üvey kardeş context."),
    ("hûncena", "hûncenayış", "draw.water-IPFV-F.SG", "V", 0.9, morph(("hûnc", "draw.water", "V"), ("en", "IPFV", "ASP"), ("a", "F.SG", "AGR")), "Su çekiyor context supports drawing/pulling water."),
    ("pwe'la", "pwe'la", "cloth", "N", 0.88, "", "Bezle bulaşık kuruluyor context supports cloth."),
    ("pêy pwe'la", "pêy pwe'la", "with a cloth", "PHRASE", 0.92, "", "Bezle contexts support with a cloth."),
    ("bweç", "bueç", "tail", "N", 0.92, "", "Animal body-part rows plus local Harun bueç/boç evidence support tail."),
    ("rezeni", "rezeni", "fennel", "N", 0.92, "", "Rezene çayı context supports fennel."),
    ("rêngnin", "reng", "colored", "ADJ", 0.9, "", "Renkli kalemler context supports colored."),
    ("qwıtida", "qwıti", "box-LOC", "N", 0.9, morph(("qwıti", "box", "N"), ("da", "LOC", "CASE")), "Konum rows use qwıtida for in/at the box."),
    ("kêy kârdiş vidyo", "kêy kârdiş vidyo", "video game", "PHRASE", 0.92, "", "Video oyunu category and image support the full phrase."),
    ("demirin", "demir", "metal/iron", "ADJ", 0.9, "", "Madeni para context supports metal/iron."),
    ("qâlb", "qâlb", "heart", "N", 0.94, "", "Kalp category/image and Arabic/Kurdish cognate evidence support heart."),
    ("sêrsibê", "sêr sıbê", "morning", "N", 0.92, "", "Sabah category/image and the existing story phrase sêr sıbê support morning."),
    ("sêrsibê şima weşib", "sêrsibê şima weşib", "good morning", "PHRASE", 0.92, "", "Günaydın greeting context supports good morning."),
    ("gwin", "guin", "blood", "N", 0.94, "", "Kan category/image plus local Harun guin evidence support blood."),
    ("barbûnyê", "barbûnye", "borlotti.bean-EZ", "N", 0.9, morph(("barbûnye", "borlotti.bean", "N"), ("ê", "EZ", "EZ")), "Barbunya context and image support borlotti bean."),
    ("baqlê barbûnyê", "baqlê barbûnyê", "borlotti bean", "PHRASE", 0.92, "", "Barbunya context supports the full phrase."),
    ("riz", "riz", "rice", "N", 0.92, "", "Pirinç and Pirinç pilavı contexts and image support rice."),
    ("rız", "riz", "rice", "N", 0.9, "", "Orthographic variant in the Pirinç category."),
    ("nûn kwêyrek", "nûn kwêyrek", "cornbread", "PHRASE", 0.92, "", "Mısır ekmeği category and image support cornbread."),
    ("yowiz", "yowiz", "also", "ADV", 0.9, "", "Conjunction rows use yowiz before an added final conjunct, meaning also/and also."),
    ("yowizik", "yowiz", "also", "ADV", 0.88, "", "Variant in conjunction rows; kept slightly lower because of the extra final consonant."),
    ("çwel", "çwel", "desert", "N", 0.9, "", "Çöl contexts support desert."),
    ("mamê", "mame", "food-EZ", "N", 0.86, morph(("mame", "food", "N"), ("ê", "EZ", "EZ")), "Bebek maması and Mama sandalyesi contexts support a baby-food/feeding compound; kept conservative because this is likely a Turkish loan."),
    ("mamê hılikû", "mamê hûlikû", "baby food", "PHRASE", 0.9, "", "Bebek maması context supports baby food."),
    ("sandalyê mamê qıcû", "sandalyê mamê qıcû", "high chair", "PHRASE", 0.9, "", "Mama sandalyesi context supports high chair."),
    ("mâqârrê la", "mâqârrê la", "thread spool", "PHRASE", 0.9, "", "Thread/yarn object context supports thread spool."),
    ("tpapê la", "tpapê la", "ball of yarn", "PHRASE", 0.9, "", "Yarn-object context supports ball of yarn."),
    ("kweçika nyûnin", "kweçika nyûnin", "dessert spoon", "PHRASE", 0.9, "", "Tatlı kaşığı category and image support dessert spoon."),
    ("kweçka nyûnina", "kweçika nyûnin", "dessert.spoon-F.SG.COP", "PHRASE", 0.88, "", "Introduction variant of the dessert-spoon phrase."),
    ("gışta nyûnin", "gışta nyûnin", "middle finger", "PHRASE", 0.9, "", "Orta parmak category and image support middle finger."),
    ("gışta mwêrika nyûnina", "gışta mwêrika nyûnin", "middle finger", "PHRASE", 0.88, "", "Introduction variant of the middle-finger phrase."),
    ("kena wırde", "wırde kerdış", "chop/cut up", "PHRASE", 0.9, "", "Doğruyor/Dilimliyor contexts and images support chopping or cutting into pieces."),
    ("ken wırde", "wırde kerdış", "chop/cut up", "PHRASE", 0.9, "", "Doğruyor context supports chopping or cutting into pieces."),
    ("dıwar wırde", "dıwar wırde", "small livestock", "PHRASE", 0.9, "", "Küçükbaş hayvanlar category and image support small livestock."),
    ("kena zwa", "zwa kerdış", "dry/wipe dry", "PHRASE", 0.9, "", "Kuruluyor and dish-drying image support drying or wiping dry."),
    ("bı zwa", "zwa bıyayış", "dried", "PHRASE", 0.88, "", "Kurudu context supports dried."),
    ("lâwê dêst", "lâwê dêst", "back of hand", "PHRASE", 0.9, "", "Elin tersi context supports back of hand."),
    ("lâwê yo binûna", "lâwê yo binûna", "on top of one another", "PHRASE", 0.88, "", "Tabakları üst üste koyuyor context supports the stacked/on-one-another phrase."),
    ("ilâc pakê", "ilâc pakê", "cleaning spray", "PHRASE", 0.9, "", "Cleaning-spray context supports the full phrase."),
    ("qâ'hryen", "qâ'hryayış", "be.sad/upset-IPFV", "V", 0.88, morph(("qâ'hr", "be.sad/upset", "V"), ("yen", "IPFV", "ASP")), "Üzüldü çünkü arabayı kırdı context supports being sad/upset."),
    ("hêysen", "hêsyayış", "look.like-IPFV", "V", 0.9, morph(("hêys", "look.like", "V"), ("en", "IPFV", "ASP")), "Bulut gibi görünüyor and similarity contexts support look like."),
    ("mûrawa", "mûra", "pear-F.SG.COP", "N", 0.88, morph(("mûra", "pear", "N"), ("wa", "F.SG.COP", "COP")), "The fruit image for Mûra is pear; standalone mûra is left ambiguous because another image uses it for bead."),
    ("hennê cındû", "hennê cındû", "lichen", "PHRASE", 0.9, "", "Linked image title Liken supports this compound."),
    ("gê cındû", "gê cındû", "stink bug", "PHRASE", 0.9, "", "Kokarca böceği context and image support stink bug."),
    ("fistix ârd", "fistix ârd", "peanut", "PHRASE", 0.9, "", "Yer fıstığı context supports peanut."),
    ("dêbûn ârd", "dêbûn ârd", "floor/ground", "PHRASE", 0.9, "", "Zemin context supports floor/ground."),
    ("pârê gawrû", "pârê gawrû", "foreign money/currency", "PHRASE", 0.86, "", "Dolar and Euro contexts suggest foreign currency; left conservative because gawrû may be a broader adjective."),
    ("qwılp bêr", "qwılp bêr", "door handle", "PHRASE", 0.9, "", "Kapı kolu context supports door handle."),
    ("şûne", "şûne", "comb", "N", 0.9, "", "Tarak image/title and baby-hair-combing context support comb."),
    ("gıcik kena şûne", "şûne kerdış", "comb hair", "PHRASE", 0.9, "", "Bebeğin saçını tarıyor image/title support combing hair."),
    ("gwıcak", "gwıcak", "button", "N", 0.92, "", "Düğme title/image support button."),
    ("qwılikê gwıcak", "qwılikê gwıcak", "buttonhole", "PHRASE", 0.92, "", "İlik title/image support buttonhole."),
    ("wıskrê", "wıskrê", "bowl/vessel", "N", 0.86, "", "Toprak kap and Hoşaf vessel contexts support bowl/vessel."),
    ("wiskrê xıl", "wıskrê xıl", "earthenware pot", "PHRASE", 0.9, "", "Toprak kap title/image support earthenware pot."),
    ("wıskrê xıl", "wıskrê xıl", "earthenware pot", "PHRASE", 0.9, "", "Surface variant for the Toprak kap phrase."),
    ("lâ'hâf", "lâ'hâf", "quilt", "N", 0.92, "", "Yorgan image/title support quilt."),
    ("tpez", "tpez", "dust", "N", 0.9, "", "Toz and Toz alıyor contexts support dust."),
    ("tpez gena", "tpez gırotış", "dust / remove dust", "PHRASE", 0.9, "", "Toz alıyor image/title support dusting."),
    ("pocena", "pocayış", "cook/bake-IPFV-F.SG", "V", 0.88, morph(("poc", "cook/bake", "V"), ("en", "IPFV", "ASP"), ("a", "F.SG", "AGR")), "Yemek yapmak and nûn pocena contexts support cooking/baking."),
    ("nûn pocena", "nûn pocayış", "bake bread", "PHRASE", 0.9, "", "Bread-making context supports bake bread."),
    ("kerrawa", "kerra", "rock-F.SG.COP", "N", 0.88, morph(("kerra", "rock", "N"), ("wa", "F.SG.COP", "COP")), "Taş/kaya contexts support the feminine/copal rock form."),
    ("âlif patik", "âlif patik", "daisy/chamomile", "PHRASE", 0.92, "", "Papatya title/image supports daisy/chamomile."),
    ("telê gwıl", "telê gwıl", "rose bush", "PHRASE", 0.92, "", "Gül çalısı title/image supports rose bush."),
    ("vâş kerrû", "vâş kerrû", "moss", "PHRASE", 0.9, "", "Yosun title/image supports moss."),
    ("vâştpûri", "vâştpûri", "sickle", "N", 0.9, "", "Orak image/title support sickle."),
    ("tpızbê", "tpızbê", "prayer beads", "N", 0.9, "", "Tesbih image/title support prayer beads."),
    ("aşma", "aşma", "moon/month", "N", 0.9, "", "Ay and Geçen ay contexts support moon/month."),
    ("aşma vyerayê", "aşma vyerayê", "last month", "PHRASE", 0.9, "", "Geçen ay context supports last month."),
    ("hewır", "hewır", "cloud", "N", 0.92, "", "Bulut image/title support cloud."),
    ("çwıli", "çwıli", "brushwood", "N", 0.9, "", "Çalı çırpı image/title support brushwood."),
    ("şak", "şak", "branch", "N", 0.9, "", "Dal image/title support branch."),
    ("şak dar", "şak dar", "tree branch", "PHRASE", 0.9, "", "Ağacın dalı context supports tree branch."),
    ("kwâtê dar", "kwâtê dar", "tree stump/log", "PHRASE", 0.9, "", "Kütük/Ağacın kütüğü image/title support tree stump or log."),
    ("qêyf keri", "qêyf keri", "cave", "PHRASE", 0.92, "", "Mağara image/title support cave."),
    ("kerrayin", "kerrayin", "rocky/stony place", "N", 0.9, "", "Taşlık image/title support rocky or stony place."),
    ("dbyaw", "dbyaw", "giant", "N", 0.9, "", "Dev image/title support giant."),
    ("astpar", "astpar", "horse", "N", 0.9, "", "At image/title support horse."),
    ("astpara", "astpar", "horse-F.SG", "N", 0.88, morph(("astpar", "horse", "N"), ("a", "F.SG", "AGR")), "Horse and unicorn contexts support the feminine/ezafe horse form."),
    ("astpara âsteryinin", "astpara âsteryinin", "unicorn", "PHRASE", 0.9, "", "Tek boynuzlu at image/title support unicorn."),
    ("sela", "sela", "basket", "N", 0.88, "", "Sepet/Sele sepeti image/title support basket; mesela is left to its own tokenization."),
    ("pwel", "pwel", "cleaning cloth", "N", 0.88, "", "Temizlik bezi image/title support cleaning cloth."),
    ("pweli masi", "pweli masi", "fish scale", "PHRASE", 0.9, "", "Balık pulu context supports fish scale."),
    ("ilâc", "ilâc", "medicine/chemical", "N", 0.86, "", "Temizlik spreyi context supports chemical/cleaning agent; gloss remains broad because ilâc can also be medicine."),
    ("kêyber", "kêyber", "garden", "N", 0.9, "", "Sebze bahçesi and watering-garden contexts support garden."),
    ("lwıl", "lwıl", "graft", "NVE", 0.88, "", "Tree aşı/grafting image supports graft."),
    ("lwıl ken", "lwıl kerdış", "graft", "PHRASE", 0.9, "", "Aşı yapıyor image with trees supports grafting."),
    ("gwıret", "gırotış", "hold.PTCP", "V", 0.86, "", "Bebeği kucağında tutuyor context supports holding; kept conservative because the exact morphology needs review."),
    ("legûnda", "legûn", "basin-LOC", "N", 0.9, morph(("legûn", "basin", "N"), ("da", "LOC", "CASE")), "Bebeği yıkıyor context with basin supports basin-LOC."),
    ("sawnena", "sawnayış", "scrub-IPFV-F.SG", "V", 0.9, morph(("sawn", "scrub", "V"), ("en", "IPFV", "ASP"), ("a", "F.SG", "AGR")), "Tezgahı ovalıyor context supports scrubbing."),
    ("tâ'hân dûn", "tâ'hân dayış", "push", "PHRASE", 0.9, "", "Arabayı itiyor image/title support push."),
    ("nırrasen", "nırrasayış", "not.reach-IPFV", "V", 0.88, morph(("nı", "NEG", "NEG"), ("rras", "reach", "V"), ("en", "IPFV", "ASP")), "Topa ulaşamıyor context supports cannot reach."),
    ("gweşt", "gweşt", "meat", "N", 0.92, "", "Et and Mangal yapıyor contexts support meat."),
    ("lâc", "lâc", "son/boy", "N", 0.88, "", "Torunun oğlu context supports son/boy component."),
    ("lâc tpêrn", "lâc tpêrn", "grandson", "PHRASE", 0.9, "", "Torunun oğlu context supports grandson."),
    ("dbirr", "dbirr", "far/distant", "ADJ", 0.88, "", "Uzaktan akraba phrase supports distant."),
    ("bi'hâq", "bi'hâq", "step-", "ADJ", 0.88, "", "Üvey baba context supports step- as a kinship modifier."),
    ("bi'hâq byâbi", "bi'hâq byâbi", "stepfather", "PHRASE", 0.9, "", "Üvey baba context supports stepfather."),
    ("qayiş", "qayiş", "belt", "N", 0.92, "", "Kemer image/title support belt."),
    ("cızmê", "cızmê", "boots", "N", 0.9, "", "Çizme image/title support boots."),
    ("lapik", "lapik", "glove", "N", 0.9, "", "Eldiven image/title support glove."),
    ("mayo", "mayo", "swimsuit", "N", 0.9, "", "Mayo image/title support swimsuit."),
    ("qârmaç", "qârmaç", "drought", "N", 0.9, "", "Kuraklık image/title support drought."),
    ("hâzaz", "hâzaz", "landslide", "N", 0.9, "", "Toprak kayması image/title support landslide."),
    ("varûn", "varûn", "rain", "N", 0.9, "", "Yağmur and rain-forecast contexts support rain."),
    ("pêl", "pêl", "wing", "N", 0.9, "", "Kanat and chicken-wing contexts support wing."),
    ("pênc", "pênc", "claw", "N", 0.9, "", "Pençe image/title support claw."),
    ("pelik", "pelik", "gill", "N", 0.9, "", "Solungaç image/title support gill."),
    ("bwım", "bwım", "owl", "N", 0.9, "", "Baykuş image/title support owl."),
    ("mwış", "mwış", "mole", "N", 0.9, "", "Köstebek image/title support mole."),
    ("mwêymwin", "mwêymwin", "monkey", "N", 0.9, "", "Maymun image/title support monkey."),
    ("naxir", "naxir", "herd/flock", "N", 0.9, "", "Koyun sürüsü context supports herd/flock."),
    ("fâkbırik", "fâkbırik", "crocodile", "N", 0.9, "", "Timsah image/title support crocodile."),
    ("çayê âlik patıkû", "çayê âlik patıkû", "chamomile tea", "PHRASE", 0.9, "", "Papatya çayı image/title support chamomile tea."),
    ("âlik patıkû", "âlik patıkû", "chamomile/daisy", "PHRASE", 0.88, "", "Papatya çayı context supports chamomile/daisy as a compound."),
    ("tyeşt", "tyeşt", "basin", "N", 0.9, "", "Leğen image/title support basin."),
    ("kwemir", "kwemir", "coal", "N", 0.9, "", "Kömür image/title support coal."),
    ("tıfıng", "tıfıng", "rifle", "N", 0.9, "", "Tüfek image/title support rifle."),
    ("pılas", "pılas", "goat-hair rug", "N", 0.9, "", "Keçi kılı kilim image/title support goat-hair rug."),
    ("çaxmax", "çaxmax", "lighter", "N", 0.9, "", "Çakmak image/title support lighter."),
    ("dêrzinê nakû", "dêrzinê nakû", "fishhook", "PHRASE", 0.9, "", "Olta iğnesi image/title support fishhook."),
    ("pırcinê ğeli", "pırcinê ğeli", "silo", "PHRASE", 0.9, "", "Silo image/title support silo."),
    ("râr", "râr", "path/road", "N", 0.9, "", "Patika and Köy yolu contexts support path/road."),
    ("râr dow", "râr dow", "village road", "PHRASE", 0.9, "", "Köy yolu context supports village road."),
    ("dow", "dow", "village", "N", 0.9, "", "Köy image/title support village."),
    ("qağıt", "qağıt", "paper", "N", 0.92, "", "Kağıt image/title support paper."),
    ("dbolavê", "dbolav", "cabinet/shelf-EZ", "N", 0.88, morph(("dbolav", "cabinet/shelf", "N"), ("ê", "EZ", "EZ")), "Kitaplık and cupboard-placement contexts support cabinet/shelf."),
    ("dbolavê kıtabû", "dbolavê kıtabû", "bookcase", "PHRASE", 0.9, "", "Kitaplık image/title support bookcase."),
    ("qâlâma", "qâlâm", "pen/pencil-EZ", "N", 0.9, morph(("qâlâm", "pen/pencil", "N"), ("a", "EZ/F.SG", "EZ")), "Kurşun kalem context supports pencil/pen component."),
    ("qâlâma qwırşin", "qâlâma qwırşin", "pencil", "PHRASE", 0.92, "", "Kurşun kalem image/title support pencil."),
    ("mâqâs", "mâqâs", "scissors", "N", 0.92, "", "Makas image/title support scissors."),
    ("sinê zerd", "sinê zerd", "greenfinch", "PHRASE", 0.9, "", "Florya bird image/title support greenfinch."),
    ("cûm", "cûm", "glass", "N", 0.88, "", "Cam material context supports glass; compounds like cûmwerd are left to separate analysis."),
    ("betûnarma", "betûnarma", "reinforced concrete", "ADJ/N", 0.9, "", "Betonarme bina image/title support reinforced concrete."),
    ("hêkim", "hêkim", "doctor", "N", 0.9, "", "Doktor image/title support doctor."),
    ("ûlax", "ûlax", "courier/messenger", "N", 0.88, "", "Profession image/title uses Ulax, likely Turkish/Ottoman ulak 'messenger'; kept conservative."),
    ("çerdek kwyû", "çerdek kwyû", "pumpkin seed", "PHRASE", 0.9, "", "Kabak çekirdeği context supports pumpkin seed."),
    ("bezrê kwyû", "bezrê kwyû", "pumpkin seed", "PHRASE", 0.9, "", "Kabak çekirdeği context supports pumpkin seed."),
    ("mêywê kwyû", "mêywê kwyû", "forest berries", "PHRASE", 0.88, "", "Orman meyveleri context supports forest berries; standalone kwyû remains unresolved because it also appears in pumpkin-seed compounds."),
    ("destê tırşik", "destê tırşik", "bunch of cress", "PHRASE", 0.88, "", "Tere image shows a bunch of greens; phrase is kept at review threshold."),
    ("dâstê tırşik", "destê tırşik", "bunch of cress", "PHRASE", 0.88, "", "Surface variant in the Tere question row."),
    ("êy", "êy", "that/they", "DEM/PRON", 0.88, "", "A.K. category rows use Êy/êy for that or plural demonstrative/pronominal reference; kept broad because it also occurs in Êy la 'for that reason'."),
    ("lâwê", "lâw", "back/upper side-EZ", "N", 0.86, morph(("lâw", "back/upper side", "N"), ("ê", "EZ", "EZ")), "Elin tersi and üst üste contexts support back/upper-side rather than the false daughter match."),
    ("destê", "dest", "bunch-EZ", "N", 0.86, morph(("dest", "bunch", "N"), ("ê", "EZ", "EZ")), "Herb rows use destê with mint, purslane, and cress where the Turkish/image context is a bunch of greens."),
    ("destê pwıni", "destê pwıni", "bunch of mint", "PHRASE", 0.9, "", "Nane image/title support bunch of mint."),
    ("destê parpar", "destê parpar", "bunch of purslane", "PHRASE", 0.9, "", "Semizotu image/title support bunch of purslane."),
    ("zwa", "zwa", "dry", "ADJ/NVE", 0.9, "", "Kurudu, Kuruluyor, and Kuru saç contexts all support dry."),
    ("zwaw", "zwa", "dry-M.SG.COP", "ADJ", 0.88, morph(("zwa", "dry", "ADJ"), ("w", "M.SG.COP", "COP")), "Kuru saç context supports the copular dry form."),
    ("wırde", "wırde", "small/pieces", "ADJ/NVE", 0.88, "", "Doğruyor/Dilimliyor and Küçükbaş contexts support small or pieces."),
    ("vırazen", "vırazenayış", "make/prepare-IPFV", "V", 0.9, morph(("vıraz", "make/prepare", "V"), ("en", "IPFV", "ASP")), "Bread, cake, and grilling contexts support make/prepare."),
    ("pocena", "pocayış", "cook/bake-IPFV-F.SG", "V", 0.9, morph(("poc", "cook/bake", "V"), ("en", "IPFV", "ASP"), ("a", "F.SG", "AGR")), "Yemek yapmak and nûn pocena contexts support cook/bake; raises earlier conservative confidence."),
    ("âlif", "âlif", "daisy/chamomile", "N", 0.88, "", "Papatya contexts support daisy/chamomile as the token in Âlif patik."),
    ("telê", "tel", "bush/stem-EZ", "N", 0.86, morph(("tel", "bush/stem", "N"), ("ê", "EZ", "EZ")), "Gül çalısı context supports bush/stem rather than the false wire/yarn match."),
    ("vâş", "vâş", "grass/herb", "N", 0.88, "", "Yosun and sickle contexts support grass/herb as a component."),
    ("gê", "gê", "bug/insect", "N", 0.86, "", "Kokarca böceği context supports bug/insect; kept broad because standalone gê needs review."),
    ("mâqârrê", "mâqârr", "spool-EZ", "N", 0.9, morph(("mâqârr", "spool", "N"), ("ê", "EZ", "EZ")), "İplik makarası context supports spool-EZ."),
    ("qwılikê", "qwılik", "hole/opening-EZ", "N", 0.88, morph(("qwılik", "hole/opening", "N"), ("ê", "EZ", "EZ")), "İlik/buttonhole context supports hole/opening-EZ."),
    ("kwâtê", "kwât", "stump/log-EZ", "N", 0.88, morph(("kwât", "stump/log", "N"), ("ê", "EZ", "EZ")), "Kütük context supports stump/log-EZ."),
    ("qêyf", "qêyf", "cave", "N", 0.9, "", "Mağara context and image support cave."),
    ("qêdna", "qêdnayış", "save/rescue.PST", "V", 0.88, "", "Kendini kurtardı context supports saved/rescued."),
    ("tâ'hân", "tâ'hân", "push", "NVE", 0.88, "", "Arabayı itiyor context supports push as the nonverbal element in tâ'hân dûn."),
    ("hârzân", "hârzân", "help", "NVE", 0.86, "", "Help context supports hârzân as part of the helping expression, but the full phrase remains the stronger analysis."),
    ("âlik", "âlik", "chamomile/daisy", "N", 0.86, "", "Papatya çayı context supports chamomile/daisy as a component."),
    ("cinyeka", "cinyek", "woman-F.SG", "N", 0.9, morph(("cinyek", "woman", "N"), ("a", "F.SG", "AGR")), "Kadın and yaşlı kadın contexts support woman-F.SG."),
    ("qıca", "qıc", "child/girl-F.SG", "N", 0.88, morph(("qıc", "child/girl", "N"), ("a", "F.SG", "AGR")), "Girl baby and baby-kissing contexts support child/girl-F.SG."),
    ("tırrm", "tırrm", "corpse/body", "N", 0.9, "", "Naaş context supports corpse/body."),
    ("hûma", "hûma", "God", "N", 0.9, "", "Good-wish formulas with Allah in the Turkish title support God."),
    ("dêrzinê", "dêrzin", "needle-EZ", "N", 0.88, morph(("dêrzin", "needle", "N"), ("ê", "EZ", "EZ")), "Olta iğnesi context supports needle-EZ."),
    ("pırcinê", "pırcin", "granary/silo-EZ", "N", 0.88, morph(("pırcin", "granary/silo", "N"), ("ê", "EZ", "EZ")), "Silo image/title support granary/silo-EZ."),
    ("dbımûn", "dbımûn", "smoke", "N", 0.9, "", "Duman image/title support smoke."),
    ("qâlâmê", "qâlâm", "pen/pencil-EZ", "N", 0.9, morph(("qâlâm", "pen/pencil", "N"), ("ê", "EZ", "EZ")), "Tahta kalemi context supports pen/pencil-EZ."),
    ("qâlâmê taxti", "qâlâmê taxti", "whiteboard marker", "PHRASE", 0.9, "", "Tahta kalemi image/title support whiteboard marker."),
    ("sinê", "sinê", "greenfinch", "N", 0.86, "", "Florya context supports greenfinch as part of Sinê zerd."),
    ("okêy", "okêy", "Okey game", "N", 0.9, "", "Okey image/title support the game."),
    ("rengli", "rengli", "colored", "ADJ", 0.9, "", "Renkli bloklar context supports colored."),
    ("kârdiş", "kerdış", "playing/doing.NMLZ", "VN", 0.88, "", "Video oyunu phrase supports playing/doing nominalization."),
    ("ârrabê", "ârrabe", "cart/car-EZ", "N", 0.88, morph(("ârrabe", "cart/car", "N"), ("ê", "EZ", "EZ")), "Alışveriş arabası context supports cart/car-EZ."),
    ("ârrabê bıgırayiş", "ârrabê bıgırayiş", "shopping cart", "PHRASE", 0.9, "", "Alışveriş arabası image/title support shopping cart."),
    ("hâşt", "hâşt", "eight", "NUM", 0.94, "", "Number category rows support eight and override the false harmonious dictionary match."),
    ("bısqılêt", "bısqılêt", "bicycle", "N", 0.9, "", "Bisiklet image/title support bicycle."),
    ("xêykel", "xêykel", "statue", "N", 0.9, "", "Heykel image/title support statue."),
    ("zıx", "zıx", "fat", "ADJ", 0.9, "", "Şişman image/title support fat."),
    ("mêtin", "mêtin", "strong", "ADJ", 0.9, "", "Güçlü image/title support strong in the adjective category."),
    ("kwırçıkın", "kwırçıkın", "skinny", "ADJ", 0.9, "", "Sıska image/title support skinny."),
    ("qâlefetın", "qâlefetın", "handsome", "ADJ", 0.9, "", "Yakışıklı image/title support handsome."),
    ("tenik", "tenik", "thin", "ADJ", 0.9, "", "İnce door adjective rows support thin."),
    ("qalm", "qalm", "thick", "ADJ", 0.9, "", "Kalın door adjective rows support thick."),
    ("qâfınayı", "qâfınayış", "locked", "ADJ/PTCP", 0.86, "", "Kilitli değil context supports locked as the adjective negated by nyû."),
    ("qâfınayı nyû", "qâfınayı nyû", "not locked", "PHRASE", 0.9, "", "Kilitli değil image/title support not locked."),
    ("hâlm", "hâlm", "steam", "N", 0.92, "", "Buhar image/title support steam."),
    ("zûn", "zûn", "tongue/language", "N", 0.9, "", "Dil body-part context supports tongue/language."),
    ("gwêş", "gwêş", "ear", "N", 0.92, "", "Kulak image/title support ear."),
    ("wâkê", "wâk", "wrist-EZ", "N", 0.88, morph(("wâk", "wrist", "N"), ("ê", "EZ", "EZ")), "Bilek context supports wrist-EZ."),
    ("wâkê dêst", "wâkê dêst", "wrist", "PHRASE", 0.9, "", "Bilek image/title support wrist."),
    ("nângûn", "nângûn", "fingernail", "N", 0.9, "", "Tırnak image/title support fingernail/nail."),
    ("wekê", "wek", "ankle/joint-EZ", "N", 0.86, morph(("wek", "ankle/joint", "N"), ("ê", "EZ", "EZ")), "Ayak bileği context supports ankle/joint-EZ."),
    ("wekê lıng", "wekê lıng", "ankle", "PHRASE", 0.9, "", "Ayak bileği image/title support ankle."),
    ("mwêrika", "mwêrik", "man-EZ/F.SG", "N", 0.86, morph(("mwêrik", "man", "N"), ("a", "EZ/F.SG", "EZ")), "Finger-name rows use mwêrika as a genitive/ezafe man/human component."),
    ("vûm", "vûm", "almond", "N", 0.9, "", "Badem image/title support almond."),
    ("hwırmê", "hwırm", "date-PL/OBL/EZ", "N", 0.9, morph(("hwırm", "date", "N"), ("ê", "PL/OBL/EZ", "AGR")), "Hurma image/title support date."),
    ("eskic", "eskic", "raisin", "N", 0.9, "", "Kuru üzüm image/title support raisin."),
    ("lêblêbi", "lêblêbi", "roasted chickpea", "N", 0.9, "", "Leblebi image/title support roasted chickpea."),
    ("katpel", "katpel", "potato", "N", 0.9, "", "Patates and Patates kızartması contexts support potato."),
    ("cow", "cow", "barley", "N", 0.9, "", "Arpa image/title support barley."),
    ("baqlê", "baqle", "bean-EZ", "N", 0.86, morph(("baqle", "bean", "N"), ("ê", "EZ", "EZ")), "Barbunya phrase supports bean-EZ."),
    ("xic", "xic", "cube", "N", 0.88, "", "Küp şeker context supports cube."),
    ("xic şekêr", "xic şekêr", "sugar cube", "PHRASE", 0.9, "", "Küp şeker image/title support sugar cube."),
    ("nâ'hê", "nâ'he", "chickpea", "N", 0.9, "", "Nohut image/title support chickpea."),
    ("mwiz", "mwiz", "banana", "N", 0.9, "", "Muz image/title support banana."),
    ("iswat", "iswat", "pepper", "N", 0.86, "", "Dolma biber context and biber image support pepper."),
    ("iswat dbalmû", "iswat dbalmû", "stuffed pepper", "PHRASE", 0.9, "", "Dolma biber image/title support stuffed pepper."),
    ("lâ'hnê", "lâ'hne", "cabbage-EZ", "N", 0.88, morph(("lâ'hne", "cabbage", "N"), ("ê", "EZ", "EZ")), "Lahana image/title support cabbage-EZ."),
    ("lâ'hnê sya", "lâ'hnê sya", "black cabbage/kale", "PHRASE", 0.9, "", "Lahana context supports cabbage phrase."),
    ("gwırâbi", "gwırâbi", "cookie", "N", 0.9, "", "Kurabiye image/title support cookie."),
    ("zêytpin", "zêytpin", "olive", "N", 0.9, "", "Zeytin image/title support olive."),
    ("kwıftê", "kwıfte", "kofta-EZ", "N", 0.86, morph(("kwıfte", "kofta", "N"), ("ê", "EZ", "EZ")), "İçli köfte context supports kofta-EZ."),
    ("kwıftê zârrû", "kwıftê zârrû", "stuffed kofta", "PHRASE", 0.9, "", "İçli köfte image/title support stuffed kofta."),
    ("kwıftê zerrû", "kwıftê zerrû", "stuffed kofta", "PHRASE", 0.88, "", "Surface variant of the İçli köfte phrase."),
    ("kwêyrek", "kwêyrek", "corn", "N", 0.88, "", "Mısır ekmeği phrase supports corn as the modifier in cornbread."),
    ("wısar", "wısar", "spring", "N", 0.9, "", "İlkbahar image/title support spring."),
    ("bât taştarê", "bât taştarê", "afternoon", "PHRASE", 0.9, "", "Öğleden sonra image/title support afternoon."),
    ("taştarê", "taştar", "noon-EZ", "N", 0.88, morph(("taştar", "noon", "N"), ("ê", "EZ", "EZ")), "Öğlen and afternoon contexts support noon-EZ."),
    ("payiz", "payiz", "autumn", "N", 0.9, "", "Sonbahar image/title support autumn."),
    ("mûryer", "mûryer", "pear tree", "N", 0.92, "", "Armut ağacı context/image support pear tree."),
    ("tpyer", "tpyer", "mulberry tree", "N", 0.9, "", "Dut ağacı context/image support mulberry tree."),
    ("darê çûm", "darê çûm", "pine tree", "PHRASE", 0.9, "", "Çam ağacı image/title support pine tree."),
    ("çûma", "çûm", "pine-F.SG", "N", 0.86, morph(("çûm", "pine", "N"), ("a", "F.SG", "AGR")), "Çam ağacı introduction rows support pine-F.SG."),
    ("sayer", "sayer", "apple tree", "N", 0.9, "", "Elma ağacı context/image support apple tree."),
    ("kabax", "kabax", "poplar tree", "N", 0.9, "", "Kavak ağacı context/image support poplar tree."),
    ("mişmişer", "mişmişer", "apricot tree", "N", 0.9, "", "Kayısı ağacı context/image support apricot tree."),
    ("zêytpyûnyer", "zêytpyûnyer", "olive tree", "N", 0.9, "", "Zeytin ağacı context/image support olive tree."),
    ("guinc", "guinc", "hawthorn", "N", 0.9, "", "Alıç context/image support hawthorn."),
    ("bêyer", "bêyer", "quince tree", "N", 0.9, "", "Ayva ağacı context/image support quince tree."),
    ("valyer", "valyer", "willow tree", "N", 0.9, "", "Söğüt ağacı context/image support willow tree."),
    ("mêr", "mêr", "hoe", "N", 0.9, "", "Çapa tool context/image support hoe."),
    ("zomp", "zomp", "sledgehammer", "N", 0.9, "", "Balyoz context/image support sledgehammer."),
    ("marrol", "marrol", "lettuce", "N", 0.9, "", "Marul image/title support lettuce."),

    # Third category-text micro-pass: clear Turkish-title/image-backed red cells.
    ("gergwış", "gergwış", "cradle", "N", 0.92, "", "Beşik title/image support cradle."),
    ("zıbınê", "zıbın", "baby undershirt-EZ", "N", 0.9, morph(("zıbın", "baby undershirt", "N"), ("ê", "EZ", "EZ")), "Zıbınê 'hılikû baby-clothing rows support baby undershirt with ezafe."),
    ("dırıkyer", "dırıkyer", "blackberry bush", "N", 0.9, "", "Böğürtlen çalısı title/image support blackberry bush."),
    ("dêrzinek", "dêrzinek", "nettle", "N", 0.9, "", "Isırgan otu title/image support nettle."),
    ("gıwêrç", "gwerç", "wild pear", "N", 0.9, "", "Yaban armudu title/image support wild pear."),
    ("filketa", "filketa", "safety pin", "N", 0.92, "", "Çengelli iğne title/image support safety pin."),
    ("fermwar", "fermwar", "zipper", "N", 0.92, "", "Fermuar title/image support zipper."),
    ("yüksük", "yüksük", "thimble", "N", 0.92, "", "Yüksük title/image support thimble."),
    ("kwele", "kwele", "firewood", "N", 0.9, "", "Odun title/image support firewood."),
    ("tic", "tic", "sun", "N", 0.92, "", "Güneş title/image support sun."),
    ("şarmyen", "şarmiyayış", "be.ashamed-IPFV", "V", 0.9, morph(("şarm", "be.ashamed", "V"), ("yen", "IPFV", "ASP")), "Utanıyor title/context support be ashamed or embarrassed."),
    ("bermen", "bermayış", "cry-IPFV", "V", 0.9, morph(("berm", "cry", "V"), ("en", "IPFV", "ASP")), "Ağlıyor and Bebek ağlıyor title/context support crying."),
    ("qıraliçe", "qıraliçe", "queen", "N", 0.92, "", "Kraliçe title/image support queen."),
    ("şâ'hmeran", "şâ'hmeran", "Shahmaran", "N", 0.9, "", "Şahmeran legend title/image support the mythic figure."),
    ("zırx", "zırx", "armor", "N", 0.9, "", "Zırh title/image support armor."),
    ("mefte", "mefte", "key", "N", 0.9, "", "Anahtar media title and Mefte rows support key."),
    ("pencıra", "pencıra", "window", "N", 0.92, "", "Pencere title/image support window."),
    ("resine", "resım", "picture", "N", 0.9, "", "Resim title/image support picture; final -e is kept in the surface form."),
    ("sâ'ât", "sâ'ât", "clock/watch", "N", 0.92, "", "Saat title/image support clock or watch."),
    ("televizyûn", "televizyûn", "television", "N", 0.92, "", "Televizyon title/image support television."),
    ("qomidin", "qomidin", "nightstand", "N", 0.9, "", "Komodin title/image support nightstand."),
    ("klima", "klima", "air conditioner", "N", 0.92, "", "Klima title/image support air conditioner."),
    ("dbılabê", "dbılab", "closet/cupboard-EZ", "N", 0.88, morph(("dbılab", "closet/cupboard", "N"), ("ê", "EZ", "EZ")), "Dbılabê kıncû / Dolabê kincû title supports cupboard or wardrobe with ezafe."),
    ("ocax", "ocax", "stove/hearth", "N", 0.9, "", "Ocak gözü context supports stove/hearth as the base noun."),
    ("sûrâ'hi", "sûrâ'hi", "pitcher", "N", 0.92, "", "Sürahi title/image support pitcher."),
    ("nyûnin", "nyûnin", "middle", "ADJ", 0.86, "", "Tatlı kaşığı and Orta parmak contexts support an intermediate/middle modifier; kept slightly lower because dessert-spoon semantics are conventional."),
    ("nyûnina", "nyûnin", "middle-F.SG", "ADJ", 0.86, morph(("nyûnin", "middle", "ADJ"), ("a", "F.SG", "AGR")), "Tatlı kaşığı and Orta parmak introduction rows support feminine middle modifier."),
    ("xıl", "xıl", "earth/clay", "N", 0.9, "", "Toprak kap context supports earth/clay."),
    ("bı'hûk pılxın", "bı'hûk pılxın", "slug", "PHRASE", 0.9, "", "Sümüklü böcek title/image support the full phrase as slug."),
    ("bû'hik hâmûmû", "bû'hik hâmûmû", "cockroach", "PHRASE", 0.9, "", "Hamam böceği title/image support cockroach."),
    ("ipegın", "ipeg", "silk-ADJ", "ADJ", 0.88, morph(("ipeg", "silk", "N"), ("ın", "ADJ", "DER")), "İpek context in silk fabric rows supports silk adjectival form."),
    ("salıncax", "salıncax", "swing", "N", 0.92, "", "Salıncak title/image support swing."),
    ("yazıci", "yazıci", "printer", "N", 0.92, "", "Yazıcı title/image support printer."),
    ("zarf", "zarf", "envelope", "N", 0.92, "", "Zarf stationery title/image support envelope."),
    ("zımba", "zımba", "stapler", "N", 0.92, "", "Zımba stationery title/image support stapler."),
    ("fıtbol", "fıtbol", "football", "N", 0.9, "", "Futbol topu title/image support football as the ball-game modifier."),
    ("vidyo", "vidyo", "video", "N", 0.9, "", "Video oyunu phrase context supports video."),
    ("banqamatik", "banqamatik", "ATM", "N", 0.92, "", "Bankamatik/ATM shopping category title supports ATM."),
    ("now", "now", "nine", "NUM", 0.92, "", "19 rows use Des now, supporting now as nine."),
    ("şeşgen", "şeşgen", "hexagon", "N", 0.92, "", "Altıgen title/image support hexagon."),
    ("hirigen", "hirigen", "triangle", "N", 0.92, "", "Üçgen title/image support triangle."),
    ("fılâmingo", "fılâmingo", "flamingo", "N", 0.92, "", "Flamingo bird title/image support flamingo."),
    ("tavûzû", "tavûz", "peacock-PL.OBL/EZ", "N", 0.88, morph(("tavûz", "peacock", "N"), ("û", "PL.OBL/EZ", "CASE")), "Tavus kuşu context supports peacock in bird compound."),
    ("qûği", "qûği", "swan", "N", 0.92, "", "Kuğu title/image support swan."),
    ("kwûnç", "kwûnç", "raven", "N", 0.9, "", "Kuzgun title/image support raven."),
    ("hep'hebek", "hep'hebek", "hoopoe", "N", 0.9, "", "İbibik title/image support hoopoe."),
    ("lâglâg", "lâglâg", "stork", "N", 0.92, "", "Lâglâg/Legleg title and bird context support stork."),
    ("delavriz", "delavriz", "titmouse", "N", 0.9, "", "Baştankara title/image support titmouse."),
    ("binawa", "bina", "building-F.SG", "N", 0.9, morph(("bina", "building", "N"), ("a", "F.SG", "AGR")), "Bina and Betonarme bina contexts support building with feminine agreement."),
    ("dındûnci", "dındûnci", "dentist", "N", 0.92, "", "Diş hekimi title/image support dentist."),
    ("postaciw", "postaci", "postman", "N", 0.9, "", "Postacı title/image support postman/postal worker."),
    ("peçete", "peçete", "napkin", "N", 0.92, "", "Peçete kitchen title/image support napkin."),
    ("tabax", "tabax", "plate", "N", 0.92, "", "Kitchen item rows and existing tabaxê analysis support plate."),
    ("qazme", "qazme", "pickaxe", "N", 0.9, "", "Qazma/qazme rural tool context supports pickaxe."),
    ("hêybe", "hêybe", "saddlebag", "N", 0.9, "", "Heybe title/image support saddlebag."),
    ("gyezı", "gyezı", "twig broom", "N", 0.88, "", "Çalı süpürgesi title/image support twig broom."),
    ("gow", "gow", "animal pen", "N", 0.9, "", "Ağıl title/image support animal pen."),
    ("zarab", "zarab", "washing place", "N", 0.88, "", "Yıkama yeri title/image support washing place."),
    ("towq", "towq", "griddle", "N", 0.9, "", "Sac rural kitchen title/image support griddle."),
    ("şalwâr", "şalwâr", "shalwar", "N", 0.9, "", "Traditional clothing title/image support shalwar trousers."),
    ("fistûn", "fistûn", "dress", "N", 0.9, "", "Traditional clothing context supports dress; overrides the unrelated adverbial dictionary match."),
    ("kılow", "kılow", "hat", "N", 0.88, "", "Seasonal clothing context supports hat/cap."),
    ("qwıcıx", "qwıcıx", "gloves", "N", 0.88, "", "Seasonal clothing context supports gloves."),
    ("sıba", "sıba", "tomorrow", "ADV", 0.94, "", "Yarın title/image support tomorrow."),
    ("sarrsya", "sarrsya", "black hair", "N", 0.9, "", "Siyah saç title/image support black hair."),
    ("hâlqê", "hâlqe", "ring-EZ", "N", 0.88, morph(("hâlqe", "ring", "N"), ("ê", "EZ", "EZ")), "Yüzük parmağı context supports ring with ezafe."),
    ("syaw", "siya", "black", "ADJ", 0.86, "", "Karaciğer and black-cabbage contexts support black as a rounded/semivowel variant of sya/siya."),

    # User correction: A.K. often has initial h- before vowel-initial stems; ê/êy is inconsistent in the data.
    ("hêsen", "êsayış", "appear-IPFV", "V", 0.96, morph(("h", "PROTH", "PHON"), ("ês", "appear", "V"), ("en", "IPFV", "ASP")), "User correction: hêsên/hêysên are appear, with A.K. prothetic h- before vowel-initial stems in other dialects."),
    ("hêsên", "êsayış", "appear-IPFV", "V", 0.96, morph(("h", "PROTH", "PHON"), ("ês", "appear", "V"), ("ên", "IPFV", "ASP")), "User correction: raised/variant spelling of hêsen appear."),
    ("hêsena", "êsayış", "appear-IPFV-F.SG", "V", 0.96, morph(("h", "PROTH", "PHON"), ("ês", "appear", "V"), ("en", "IPFV", "ASP"), ("a", "F.SG", "AGR")), "User correction: feminine singular imperfective form of appear."),
    ("hêysen", "êysayış", "appear-IPFV", "V", 0.96, morph(("h", "PROTH", "PHON"), ("êys", "appear", "V"), ("en", "IPFV", "ASP")), "User correction: êy spelling variant of the appear stem."),
    ("hêysên", "êysayış", "appear-IPFV", "V", 0.96, morph(("h", "PROTH", "PHON"), ("êys", "appear", "V"), ("ên", "IPFV", "ASP")), "User correction: raised êy spelling variant of appear."),
    ("hêysena", "êysayış", "appear-IPFV-F.SG", "V", 0.96, morph(("h", "PROTH", "PHON"), ("êys", "appear", "V"), ("en", "IPFV", "ASP"), ("a", "F.SG", "AGR")), "User correction: feminine singular êy spelling variant of appear."),
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_existing_keys() -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    if not CORRECTIONS_PATH.exists():
        return keys
    with CORRECTIONS_PATH.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row.get("source") == SOURCE:
                keys.add((normalize_key(row.get("form", "")), row.get("gloss_en", "")))
    return keys


def occurrence_index() -> dict[str, dict[str, object]]:
    occurrences: dict[str, dict[str, object]] = defaultdict(lambda: {"count": 0, "lesson_ids": set(), "line_ids": []})
    for path in OUTPUT_DIR.glob("ak_cat_*.tokens.tsv"):
        with path.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                key = normalize_key(row.get("form", ""))
                rec = occurrences[key]
                rec["count"] = int(rec["count"]) + 1
                rec["lesson_ids"].add(row.get("lesson_id", ""))
                line_id = row.get("line_id", "")
                if line_id and line_id not in rec["line_ids"]:
                    rec["line_ids"].append(line_id)
    return occurrences


def correction_rows() -> list[dict[str, str]]:
    now = utc_now()
    occurrences = occurrence_index()
    rows: list[dict[str, str]] = []
    for form, lemma, gloss, pos, confidence, morphemes, note in RAW_CORRECTIONS:
        key = normalize_key(form)
        rec = occurrences.get(key, {"count": 0, "lesson_ids": set(), "line_ids": []})
        lesson_ids = sorted(rec["lesson_ids"])
        rows.append(
            {
                "form": form,
                "corrected_form": "",
                "lemma": lemma,
                "gloss_en": gloss,
                "gloss_tr": "",
                "pos": pos,
                "confidence": str(confidence),
                "source": SOURCE,
                "note": f"{note} Occurrences in generated category texts before this pass: {rec['count']}.",
                "status": "approved",
                "lesson_ids": ",".join(lesson_ids) if len(lesson_ids) <= 8 else "ak_category_texts",
                "line_ids": ",".join(str(item) for item in rec["line_ids"][:12]),
                "updated_at": now,
                "morphemes": morphemes,
            }
        )
    return rows


def upsert_source_rows(rows: list[dict[str, str]]) -> tuple[int, int]:
    incoming = {(normalize_key(row["form"]), row["gloss_en"]): row for row in rows}
    if not CORRECTIONS_PATH.exists():
        with CORRECTIONS_PATH.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, delimiter="\t", fieldnames=FIELDS)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in FIELDS})
        return len(rows), 0

    with CORRECTIONS_PATH.open(encoding="utf-8-sig", newline="") as f:
        existing_rows = list(csv.DictReader(f, delimiter="\t"))

    seen: set[tuple[str, str]] = set()
    updated_rows: list[dict[str, str]] = []
    updated_count = 0
    for row in existing_rows:
        key = (normalize_key(row.get("form", "")), row.get("gloss_en", ""))
        replacement = incoming.get(key)
        if row.get("source") == SOURCE and replacement is not None:
            normalized = {field: replacement.get(field, "") for field in FIELDS}
            current = {field: row.get(field, "") for field in FIELDS}
            if current != normalized:
                updated_count += 1
            updated_rows.append(normalized)
            seen.add(key)
        else:
            updated_rows.append({field: row.get(field, "") for field in FIELDS})

    new_rows = [row for key, row in incoming.items() if key not in seen]
    updated_rows.extend({field: row.get(field, "") for field in FIELDS} for row in new_rows)

    with CORRECTIONS_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(updated_rows)
    return len(new_rows), updated_count


def write_audit(rows: list[dict[str, str]]) -> None:
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    audit_fields = ["form", "lemma", "gloss_en", "pos", "confidence", "lesson_ids", "line_ids", "note", "morphemes"]
    with AUDIT_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=audit_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in audit_fields})


def main() -> int:
    rows = correction_rows()
    added, updated = upsert_source_rows(rows)
    write_audit(rows)
    print(json.dumps({"candidate_rows": len(rows), "added_rows": added, "updated_rows": updated, "audit": str(AUDIT_PATH)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
