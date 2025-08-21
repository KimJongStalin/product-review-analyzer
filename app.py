# app.py (版本 4.0 - 采用“基础+覆写”画像逻辑)

import streamlit as st
import pandas as pd
import json
import io
from review_analyzer_core import ReviewAnalyzer # 确保 review_analyzer_core.py 在同一文件夹

@st.cache_resource
def setup_nltk_resources():
    print("正在下載 NLTK 資源...")
    nltk.download('punkt_tab')
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    print("NLTK 資源下載完畢。")

# 在應用程式執行之初就調用設定函數
setup_nltk_resources()
# --- 页面基础设置 ---
st.set_page_config(page_title="产品评论自动分析报告", layout="wide")
st.title("🚀 全功能产品评论分析报告生成器")
st.markdown("欢迎使用！请在左侧边栏完成设置，然后点击“开始生成报告”按钮。")

# --- 核心配置: “基础”与“覆写”画像 ---

# 1. 定义“基础”关键词，这是适用于所有产品的通用分析规则。
BASE_FEATURE_KEYWORDS = {
    # ===== 1. 颜色种类 =====
         '颜色种类': {
        '正面-色彩丰富': ['many colors', 'lot of colors', 'plenty of colors', 'good range', 'great variety', 'great selection', 'every color', 'all the colors', 'so many options'],
        '负面-色彩单调/反馈': ['limited range', 'not enough colors', 'wish for more', 'missing colors', 'disappointed with selection', 'needs more colors'],
        '正面-套装/数量选择满意': ['love the large set', 'great number of colors', 'perfect amount of colors', 'huge set of 72', 'full set is amazing', 'good assortment'],
        '负面-套装/数量选择不满意': ['wish for a smaller set', 'too many colors', 'no smaller option', 'forced to buy the large set', 'have to buy the whole set'],
        '正面-色系规划满意': ['great color selection', 'perfect pastel set', 'good range of skin tones', 'well-curated palette','love the color story', 'beautiful assortment of colors', 'has every color I need'],
        '负面-色系规划不满': ['missing key colors', 'no true red', 'needs more grays', 'too many similar colors','palette is not useful', 'wish it had more pastels', 'poor color selection', 'needs more skin tones'],
        },

    # ===== 2. 色彩一致性 =====
        '色彩一致性': {
        '正面-颜色准确': ['true to color', 'match the cap', 'accurate color', 'color accuracy', 'exact color', 'matches perfectly', 'consistent color', 'consistency'],
        '负面-颜色偏差': ['inconsistent', 'different shade', 'not the same', 'misleading cap', 'cap is wrong', 'color is off', 'darker than cap', 'lighter than cap', 'doesn\'t match', 'wrong color'],
        '正面-设计-颜色准确 (VS 笔帽)': ['true to color', 'match the cap', 'matches the cap perfectly', 'cap is a perfect match', 'cap is accurate'],
        '负面-设计-颜色误导 (VS 笔帽)': ['misleading cap', 'cap is wrong', 'cap is a lie', 'color doesn\'t match the barrel','the cap color is way off', 'nothing like the cap'],
        '正面-营销-颜色准确(VS 网图)': ['exactly as advertised', 'what you see is what you get', 'matches the online photo', 'true to the swatch', 'photo is accurate'],
        '负面-营销-图片误导 (VS 网图)': ['looks different from the online swatch', 'not the color in the picture', 'misrepresented color','photo is misleading', 'swatch card is inaccurate'],
        '正面-生产-品控(VS 其他笔)': ['consistent color', 'consistency', 'no variation between pens', 'reliable color', 'batch is consistent'],
        '负面-生产-品控偏差(VS 其他笔)': ['inconsistent batch', 'color varies from pen to pen', 'my new pen is a different shade', 'no quality control', 'batch variation'],
        },
    # ===== 3. 色彩饱和度与混合 =====
        '色彩饱和度与混合': {
        '正面-鲜艳/饱和': ['bright colors', 'nice and bright','beautifully bright','richly saturated', 'perfectly saturated', 'deeply saturated','nice saturation', 'vibrant colors', 'rich colors','colors pop'],
        '负面-太鲜艳/刺眼': ['garish colors', 'colors are too loud','too neon', 'too bright', 'too fluorescent', 'overly bright'],
        '负面-暗淡/褪色': ['dull', 'faded', 'pale', 'washed out', 'not bright', 'too pale', 'lackluster'
                  'colors are too dull', 'too pale', 'lackluster', 'muddy colors', 'colors look dirty', 'desaturated'],
        '正面-易于混合/渐变好': ['easy to blend', 'blends well', 'blendable', 'effortless blending', 'seamless blend', 'smooth gradient', 'layers nicely', 'buildable color', 'reactivate with water'],
        '负面-混合效果差': ['difficult to blend', 'hard to blend', 'doesn\'t blend', 'impossible to blend', 'gets muddy', 'pills paper', 'damages paper', 'dries too fast to blend', 'lifts ink'],
        },

        '色系评价': {
        '正面-喜欢标准/基础色系': ['good standard colors', 'love the basic set', 'has all the primary colors', 'classic colors'],
        '正面-喜欢鲜艳/饱和色系': ['love the vibrant colors', 'bright colors', 'bold colors', 'rich colors', 'vivid colors','highly saturated', 'nicely saturated', 'colors are saturated',
                      'colors pop', 'really pop', 'makes the colors pop'],
        '正面-喜欢粉彩色/柔和系': ['love the pastel colors', 'soft colors', 'subtle shades', 'mild colors', 'macaron colors', 'beautiful pastels','unlike neon','unlike fluorescent'
                    'non-neon', 'not neon', 'soft colors', 'subtle shades', 'not bright', 'not fluorescent', 'mild colors', 'muted tones'],
        '正面-喜欢复古/怀旧色系': ['love the vintage colors', 'retro palette', 'muted tones', 'nostalgic colors', 'old school colors'],
        '正面-喜欢莫兰迪色系': ['love the morandi colors', 'dusty colors', 'grayish tones', 'muted and elegant', 'sophisticated colors'],
        '正面-喜欢中性/肤色系': ['great range of skin tones', 'perfect neutral palette', 'good beiges', 'useful for portraits', 'love the skin tones'],
        '正面-喜欢大地/自然色系': ['love the earth tones', 'natural colors', 'beautiful botanical colors', 'forest greens', 'desert tones', 'ocean blues'],
        '正面-喜欢灰色系': ['love the gray scale', 'great set of cool grays', 'perfect warm grays', 'good neutral grays'],
        '正面-喜欢季节/主题色系': ['beautiful forest colors', 'love the ocean tones', 'perfect autumn palette', 'spring colors set', 'nice seasonal set'],
        '正面-喜欢霓虹/荧光色系': ['love the neon colors', 'like the bright fluorescent colors', 'neon pops', 'vibrant neon','beautiful neon colors'],
        '正面-喜欢金属/珠光色系': ['love the metallic colors', 'great metallic effect', 'nice metallic sheen', 'shiny metal finish', 'beautiful chrome finish', 'looks like real metal',
                      'love the pearlescent finish', 'nice shimmer'],
        '负面-色系搭配不佳': ['palette is ugly', 'colors don\'t go well together', 'weird color combination', 'unusable colors in set', 'poorly curated'],
        },


    # ===== 4. 笔头 =====
        '笔头表现': {
        '正面-双头设计认可': ['love the dual tip', 'like the dual tip', 'useful dual tip', 'handy dual tip', 'versatile design', 'great having two tips', 'love that it has two sides'],
        '负面-双头设计抱怨': ['useless dual tip', 'redundant dual tip', 'unnecessary dual tip', 'don\'t need the dual tip', 'never use the other side'],
        '正面-软头表现好': ['love the brush tip', 'flexible brush', 'great brush nib', 'smooth brush'],
        '负面-软头表现差': ['brush tip frays', 'brush tip split', 'mushy brush tip', 'brush tip wore out', 'inconsistent brush line'],
        '正面-细头表现好': ['love the fine tip', 'great for details', 'precise fine liner', 'crisp fine lines'],
        '负面-细头表现差': ['fine tip is scratchy', 'fine tip dried out', 'bent the fine tip', 'fine tip broke', 'inconsistent fine line'],
        '正面-凿头表现好': ['chisel tip is great', 'good for highlighting', 'sharp chisel edge'],
        '负面-凿头表现差': ['chisel tip is too broad', 'chisel tip wore down', 'dull chisel tip'],
        '正面-圆头表现好': ['bullet tip is sturdy', 'consistent bullet nib', 'good for writing'],
        '负面-圆头表现差': ['bullet tip skips', 'bullet nib is dry', 'wobbly bullet tip'],
        '正面-弹性好/软硬适中': ['flexible', 'great flexibility', 'nice spring', 'good snap', 'bouncy tip', 'soft brush'],
        '负面-过软/过硬/无弹性': ['too stiff', 'too firm', 'too soft', 'no flexibility', 'mushy', 'hard to control flex'],
        '正面-笔尖可替换': ['replaceable nibs', 'can replace the tips', 'interchangeable tips', 'love the replacement nibs'],
        '负面-笔尖不可替换': ['wish the tips were replaceable', 'can\'t replace the nib', 'no replacement nibs'],
        '正面-软头(Brush)-粗细变化好': ['good line variation', 'can make thick and thin lines', 'great control over stroke width', 'responsive brush'],
        '负面-软头(Brush)-粗细难控': ['hard to get a thin line', 'only makes thick strokes', 'inconsistent line width', 'no line variation'],
        '正面-细头(Fine)-粗细适合细节': ['perfect for details', 'love the 0.4mm fine tip', 'thin enough for writing', 'great for fine lines', 'super fine point'],
        '负面-细头(Fine)-粗细不合适': ['too thick for a fine liner', 'not a true 0.3mm', 'wish it was thinner', 'still too broad for small spaces'],
        '正面-凿头(Chisel)-宽度合适': ['perfect width for highlighting', 'good broad edge', 'nice thick lines for headers'],
        '负面-凿头(Chisel)-宽度不合适': ['too wide for my bible', 'too narrow for a highlighter', 'chisel tip is too thick'],
        '正面-圆头(Bullet)-粗细均匀': ['nice medium point', 'consistent line width', 'good for coloring', 'reliable bullet tip'],
        '负面-圆头(Bullet)-粗细问题': ['bullet tip is too bold', 'not a medium point as advertised'],

       },

    # ===== 5. 笔头耐用性 =====
        '笔头耐用性': {
        '正面-坚固/保形': [ 'durable tip', 'sturdy', 'robust', 'long lasting tip', 'heavy duty', 'resilient', 'holds up well', 'retains shape', 'holds its point', 'keeps its point', 'point stays sharp',  'doesn\'t get mushy', 'doesn\'t go flat',  'doesn\'t fray', 'no fraying', 'no splitting', 'resists fraying'  ],
        '负面-磨损/分叉': ['fray', 'fraying', 'frayed tip', 'split', 'splitting', 'split nib',  'wear out', 'wear down', 'wore out fast', 'tip wear', 'fell apart', 'disintegrated', 'unraveled', 'tip became fuzzy', 'fibers came apart'],
        '负面-形变/软化': ['gets mushy', 'too soft', 'tip softened', 'spongy tip', 'loses its point', 'lost its fine point', 'point went dull', 'no longer sharp', 'deformed', 'lose its shape', 'went flat', 'lost its snap', 'doesn\'t spring back'],
        '负面-意外损坏': ['bent tip', 'breaks easily', 'snapped', 'snapped off', 'cracked tip', 'chipped tip', 'broke', 'broken', 'damaged tip', 'tip fell out', 'pushed the tip in', 'tip receded'],
        '负面-寿命不匹配': ['tip wore out before ink ran out', 'felt tip died before the ink', 'plenty of ink left but tip is useless', 'tip dried out but pen is full','nib is gone but still has ink']
        },

    # ===== 6. 流畅性 (流畅性) =====
        '流畅性': {
        '正面-书写流畅': ['smooth', 'smoothness', 'glide', 'flow', 'consistent ink', 'juicy', 'wet', 'writes well', 'no skipping'],
        '负面-干涩/刮纸/断墨': ['scratchy', 'dry', 'skip', 'skipping', 'hard start', 'dried up', 'inconsistent flow', 'stops writing'],
        '负面-出墨过多/漏墨': ['blotchy', 'too much ink', 'too wet', 'leaks'],
        '正面-防渗透/防鬼影': ['no bleed', 'not bleed', 'doesn\'t bleed', 'minimal bleed', 'no ghosting', 'zero ghosting'],
        '负面-渗透/鬼影问题': ['bleed', 'ghost', 'bleed-through', 'ghosting', 'show-through', 'bleeds through', 'ghosts badly', 'feathering'],
        },

    # ===== 7. 墨水特性 (原墨水质量, 干燥速度等) =====
        '墨水特性': {
        '正面-干燥快/防涂抹': ['quick dry', 'fast dry', 'dries quickly', 'no smear', 'no smudge', 'smear proof', 'smudge proof', 'good for lefties'],
        '负面-干燥慢/易涂抹': ['smear', 'smudge', 'smears easily', 'smudges', 'takes forever to dry', 'not for left-handed'],
        '正面-环保/安全/无味': ['non-toxic', 'acid-free', 'safe for kids', 'archival', 'no smell', 'odorless', 'low odor'],
        '负面-气味难闻': ['odor', 'smell', 'fumes', 'chemical smell', 'strong smell', 'toxic smell', 'bad smell'],
        '正面-持久/防水': ['waterproof', 'water resistant', 'fade proof', 'fade resistant', 'lightfast', 'permanent', 'long lasting ink'],
        '负面-易褪色/不防水': ['not permanent', 'fades quickly', 'washes away', 'not waterproof'],
        '正面-续航长': ['longevity', 'last long', 'lasted a long time', 'plenty of ink'],
        '负面-消耗快': ['run out', 'run dry', 'dries out', 'died quickly', 'empty fast', 'no ink', 'used up too fast'],
        '正面-金属效果好': ['great metallic effect', 'nice metallic sheen', 'shiny metal finish','strong metallic look', 'looks like real metal', 'beautiful chrome finish', 'very reflective'],
        '负面-金属效果差': ['dull metallic', 'not shiny', 'no metallic effect', 'looks flat', 'weak sheen', 'not reflective'],
        '正面-闪光效果好': ['lots of glitter', 'beautiful shimmer', 'sparkly', 'glitter is vibrant','nice pearlescent effect', 'very glittery', 'good sparkle'],
        '负面-闪光效果差': ['not enough glitter', 'no shimmer', 'glitter falls off', 'dull sparkle','barely any glitter', 'messy glitter'],
        '正面-荧光/霓虹效果好': ['neon pops', 'very bright neon', 'glows under blacklight', 'super fluorescent', 'vibrant neon','glows nicely'],
        '负面-荧光/霓虹效果淡': ['neon is dull', 'not very bright', 'doesn\'t glow', 'not a true neon color','disappointing neon'],
        '负面-荧光/霓虹效果过饱和': ['too neon', 'too bright', 'too fluorescent', 'too neon/bright'],
        '正面-变色效果好': ['love the color change', 'chameleon effect is stunning', 'shifts colors beautifully', 'works in the sun', 'heat sensitive works'],
        '负面-变色效果差': ['doesn\'t change color', 'color shift is weak', 'barely changes', 'no chameleon effect'],
        '正面-夜光效果好': ['glows brightly in the dark', 'long lasting glow', 'charges quickly', 'very luminous'],
        '负面-夜光效果差': ['doesn\'t glow', 'glow is weak', 'fades too fast', 'barely glows'],
        '正面-香味好闻': ['smells great', 'love the scent', 'nice fragrance', 'fun scents', 'smells like fruit'],
        '负面-香味难闻/太浓': ['smell is too strong', 'bad smell', 'doesn\'t smell like anything', 'chemical smell', 'artificial scent'],
        '正面-可擦除效果好': ['erasable', 'erases cleanly', 'erases completely', 'no ghosting after erasing', 'frixion works well'],
        '负面-可擦效果差': ['doesn\'t erase', 'leaves a stain', 'smears when erased', 'damages paper when erasing', 'hard to erase'],

        },

    # ===== 8. 笔身与易用性 (原笔体材质, 体验等) =====
        '笔身与易用性': {
        '正面-材质/做工好': ['durable body', 'sturdy', 'well-made', 'solid', 'quality feel', 'feels premium'],
        '负面-材质/做工差': ['feels cheap', 'flimsy', 'crack', 'break', 'cheap plastic', 'broke when dropped'],
        '正面-握持舒适': ['comfortable', 'comfort', 'ergonomic', 'nice to hold', 'well-balanced', 'good grip', 'feels good in hand'],
        '负面-握持不适': ['uncomfortable', 'awkward', 'fatigue', 'cramp', 'hurts hand', 'too thick', 'too thin', 'slippery'],
        '正面-笔帽体验好': ['cap posts well', 'secure fit', 'airtight', 'cap clicks', 'easy to open cap'],
        '负面-笔帽体验差': ['hard to open cap', 'loose cap', 'cap falls off', 'cap doesn\'t stay on', 'cracked cap', 'cap broke'],
        '正面-易于使用/便携': ['easy to use', 'convenient', 'handy', 'intuitive', 'portable', 'travel', 'on the go', 'compact']
         },
        '绘画表现': {
        '正面-线条表现好/可控': ['good control', 'controllable lines', 'great line variation', 'crisp lines', 'consistent lines', 'clean lines', 'no skipping', 'sharp lines', 'great for fine details'],
        '负面-线条表现差/难控': ['hard to control', 'inconsistent line', 'uncontrollable', 'not for details', 'wobbly lines', 'shaky lines', 'broken line'],
        '正面-覆盖力好/不透明': ['opaque', 'good coverage', 'covers well', 'one coat', 'hides underlying color', 'works on dark paper', 'great opacity'],
        '负面-过于透明/覆盖力差': ['not opaque', 'too sheer', 'doesn\'t cover', 'needs multiple coats', 'transparent', 'see through'],
        '正面-涂色均匀': ['even application', 'smooth application', 'no streaks', 'self-leveling', 'consistent color', 'no streaking'],
        '负面-涂色不均': ['streak', 'streaky', 'streaking', 'leaves streaks', 'patchy', 'blotchy'],
        '正面-可再激活': ['reactivate with water', 'lifts easily for effects', 'movable ink', 'good workable time', 'can be reactivated'],
        '负面-不可再激活/易损坏': ['doesn\'t reactivate', 'lifts unintentionally', 'smears when layered', 'dries too permanent'],
        '正面-兼容铅笔': ['goes over pencil cleanly', 'doesn\'t smudge graphite', 'erases pencil underneath', 'covers pencil lines well'],
        '负面-铅笔兼容性差': ['smears pencil lines', 'smudges graphite', 'lifts graphite', 'muddy with pencil', 'doesn\'t cover pencil'],
        '正面-兼容勾线笔': ['doesn\'t smear fineliner', 'works with micron pens', 'layers over ink', 'copic-proof ink compatible', 'safe over ink'],
        '负面-勾线笔兼容性差': ['smears fineliner ink', 'reactivates ink', 'lifts the ink line', 'bleeding with ink lines', 'makes ink run'],
        '正面-兼容水彩/水粉': ['layers over watercolor', 'works well with gouache', 'can use for watercolor effects', 'doesn\'t lift watercolor'],
        '负面-水彩/水粉兼容性差': ['lifts watercolor', 'muddy with gouache', 'reactivates paint underneath', 'smears watercolor'],
        '正面-兼容彩铅': ['layers well with colored pencils', 'good for marker and pencil', 'blends with pencil crayon', 'works over wax pencil'],
        '负面-彩铅兼容性差': ['waxy buildup with colored pencils', 'doesn\'t layer over pencil crayon', 'reacts weirdly with other markers'],
        '负面-不兼容彩铅': ['waxy buildup with colored pencils', 'doesn\'t layer over pencil crayon', 'smears the pencil wax'],
        '正面-兼容酒精性马克笔': ['blends with other alcohol markers', 'works with my copics', 'blends with ohuhu', 'good Copic alternative', 'matches Copic colors', 'layers well with alcohol ink', 'smooth blend with other brands'],
        '负面-不兼容酒精性马克笔': ['doesn\'t blend with copics', 'reacts with other alcohol inks', 'smears when layered with alcohol markers', 'color matching is off', 'leaves a weird texture'],
        '正面-兼容水性马克笔': ['layers well with water-based', 'works with Tombows', 'doesn\'t reactivate water based ink', 'great for highlighting over Tombow', 'doesn\'t smear my Mildliners', 'good for underpainting'],
        '负面-不兼容水性马克笔': ['doesn\'t blend with tombows', 'smears my Mildliners', 'makes water based ink bleed', 'reactivates my tombows', 'makes a muddy mess with water-based'],
        '正面-兼容丙烯马克笔': ['layers nicely over Posca', 'can draw on top of Posca', 'doesn\'t lift the acrylic', 'good with acrylic markers', 'adheres well to paint'],
        '负面-不兼容丙烯马克笔': ['smears Posca paint', 'doesn\'t stick to acrylic marker', 'lifts the underlying acrylic', 'scratches off the acrylic surface'],
        },
        '场景表现': {
        '正面-适合大面积填色': ['great for coloring', 'good for large areas', 'fills spaces evenly', 'no streaking in large blocks', 'coloring book friendly', 'smooth coverage'],
        '负面-不适合大面积填色': ['streaky when coloring', 'dries too fast for large areas', 'bad for filling large spaces', 'leaves marker lines', 'patchy on large areas'],
        '正面-适合漫画/动漫创作': ['great for manga', 'perfect for comics', 'blends skin tones beautifully', 'works for anime style', 'good for cel shading', 'great for character art'],
        '负面-不适合漫画/动漫创作': ['hard to blend skin tones', 'colors aren\'t right for manga', 'smears my line art', 'not good for comic art'],
        '正面-适合插画创作': ['great for illustration', 'professional illustration results', 'layers beautifully for art', 'vibrant illustrations', 'perfect for artists'],
        '负面-不适合插画创作': ['not for professional illustration', 'colors are not vibrant enough for art', 'muddy blends for illustration', 'hobby grade only'],
        '正面-适合工业/产品设计': ['great for industrial design', 'perfect for rendering', 'flat even color for design', 'good for product sketches', 'excellent range of grays for design'],
        '负面-不适合工业/产品设计': ['streaky for rendering', 'colors are not suitable for design', 'not precise enough for product design', 'needs more neutral grays'],
        '正面-适合手账/日记': ['perfect for journaling', 'great for planners', 'no bleed in my hobonichi', 'mild colors are great for bujo', 'excellent for bible journaling'],
        '负面-不适合手账/日记': ['bleeds through journal pages', 'ghosts too much for planners', 'colors are too bright for journaling', 'ruined my leuchtturm'],
        '正面-适合着色书/填色': ['great for coloring books', 'perfect for adult coloring', 'coloring book friendly','no bleed in coloring book', 'doesn\'t ghost on coloring pages', 'safe for single-sided books',
                     'fine tip is perfect for intricate designs', 'great for mandalas', 'gets into tiny spaces'],
        '负面-不适合着色书/填色': ['not for coloring books', 'ruined my coloring book', 'bleeds through every page', 'ghosting is too bad for coloring books', 'ruined the next page', 'tip is too broad for detailed coloring', 'bleeds outside the lines in small patterns','pills the coloring book paper', 'tears the paper'],
        '正面-适合书法/手写艺术': ['perfect for calligraphy', 'great for hand lettering', 'nice thick and thin strokes','good for upstrokes and downstrokes', 'flexible tip for lettering', 'rich black for calligraphy'],
        '负面-不适合书法/手写艺术': ['tip is too stiff for calligraphy', 'hard to control line variation', 'ink feathers during lettering','not good for brush lettering', 'ink is not dark enough for calligraphy'],
        '正面-适合思维导图/视觉笔记': ['perfect for mind mapping', 'great for sketchnotes', 'ideal for visual notes', 'colors are bright for diagrams', 'no bleed on my notebook', 'multiple tip sizes are useful'],
        '负面-不适合思维导图/视觉笔记': ['bleeds through note paper', 'colors are too dull for charts', 'tip is too broad for visual notes'],
        '正面-适合手工艺/物品定制': ['great for diy projects', 'perfect for customizing shoes', 'works on canvas bags', 'permanent on rocks and wood', 'good for crafting'],
        '负面-不适合手工艺/物品定制': ['wipes off from plastic', 'not for outdoor use', 'color fades on fabric', 'doesn\'t work on sealed surfaces'],
        '正面-适合儿童/教学': ['great for kids', 'safe for children', 'non-toxic', 'washable ink', 'durable tip for heavy hands', 'bright colors for kids', 'good for classroom use'],
        '负面-不适合儿童/教学': ['strong smell not for kids', 'ink stains clothes', 'tip broke easily with pressure', 'cap is hard for a child to open'],

        },

        '表面/介质表现': {
        '正面-在专业纸张上表现好': ['works great on marker paper', 'smooth on bristol board', 'doesn\'t pill watercolor paper','blends well on bleedproof paper', 'perfect for mixed media paper'],
        '负面-在专业纸张上表现差': ['still bleeds through marker paper', 'feathers on hot press paper', 'destroys bristol surface', 'pills my cold press paper', 'doesn\'t blend on this paper'],
        '正面-在深色纸张上显色好': ['opaque on black paper', 'shows up well on dark paper', 'great coverage on kraft paper','vibrant on colored paper', 'pops on black', 'shows up beautifully'],
        '负面-在深色纸张上显色效果差': ['not opaque on black', 'disappears on dark paper', 'too transparent for colored paper','doesn\'t show up', 'color looks dull on black'],
        '正面-在光滑表面附着力好': ['writes on glass', 'permanent on plastic', 'adheres to metal', 'dries on ceramic', 'doesn\'t wipe off', 'great for glossy photos', 'works on whiteboards'],
        '负面-在光滑表面附着力差': ['wipes off glass', 'scratches off plastic', 'smears on metal', 'never dries on ceramic','beads up on the surface', 'poor adhesion', 'not for non-porous surfaces'],
        '正面-在布料上效果好': ['great on fabric', 'doesn\'t bleed on canvas', 'permanent on t-shirt', 'holds up in the wash','vibrant on textile', 'perfect for customizing shoes', 'doesn\'t feather on cotton'],
        '负面-在布料上效果差': ['bleeds on fabric', 'feathers on canvas', 'fades after washing', 'washes out', 'makes the fabric stiff', 'not for denim'],
        '正面-在木材上表现好': ['great on wood', 'soaks in evenly', 'vibrant color on wood', 'dries nicely on wood', 'perfect for wood crafts', 'doesn\'t bleed with the grain', 'sharp lines on wood'],
        '负面-在木材上表现差': ['bleeds into the wood grain', 'soaks in too much', 'color looks dull on wood', 'uneven color on wood', 'smears on sealed wood', 'makes the wood grain swell'],
        '正面-在石头上表现好': ['great for rock painting', 'vibrant on rocks', 'opaque on stone', 'doesn\'t scratch off easily', 'smooth lines on rocks', 'durable on pebbles'],
        '负面-在石头上表现差': ['scratches off rocks', 'not opaque enough for stone', 'color is dull on rocks', 'clogs tip on rough stone', 'hard to draw on rocks', 'fades on stone'],
        '正面-在粘土上表现好': ['works on polymer clay', 'great on air dry clay', 'doesn\'t react with sealant', 'vibrant on clay', 'soaks in nicely on bisque'],
        '负面-在粘土上表现差': ['doesn\'t adhere to clay', 'smears on polymer clay', 'reacts with the varnish', 'clogs tip on un-sanded clay'],
        '正面-在卡纸上表现好': ['great on cardstock', 'perfect for cardstock', 'no bleed on cardstock', 'vibrant on heavy paper', 'dries fast on cardstock', 'smooth on cardstock'],
        '负面-在卡纸上表现差': ['bleeds through cardstock', 'ghosting on cardstock', 'pills my cardstock','smears on glossy cardstock', 'feathers on cardstock', 'dries too slowly on cardstock'],
        '正面-兼容印台/图章': ['great for coloring stamped images', 'doesn\'t smear stamp ink', 'works with memento ink', 'no bleed lines', 'alcohol-proof ink', 'safe for stamping'],
        '负面-不兼容印台/图章': ['smears my stamp ink', 'reactivates the stamp pad ink', 'makes the lines muddy','smudges my versafine ink', 'lifts the stamp ink'],
        '正面-适合刻字/细节': ['perfect for lettering', 'great for calligraphy', 'nice for writing greetings','fine tip for small details', 'beautiful for sentiments'],
        '负面-不适合刻字/细节': ['too thick for lettering', 'bleeds when writing', 'hard to do calligraphy with'],
        },


        # ===== 9. 外观与包装 (保留) =====
        '外观与包装': {
        '正面-外观/设计美观': ['beautiful design', 'pretty', 'stylish', 'minimalist', 'sleek', 'cute', 'lovely', 'gorgeous', 'aesthetic'],
        '负面-外观廉价/丑': ['looks cheap', 'looks like a toy', 'toy-like', 'ugly'],
        '正面-包装美观/保护好': ['beautiful packaging', 'nice packaging', 'giftable', 'well packaged', 'arrived safe', 'sturdy case', 'tin case', 'reusable case'],
        '负面-包装廉价/易损坏': ['flimsy packaging', 'damaged box', 'broken case', 'arrived damaged', 'cheap case'],
        '正面-收纳便利': ['well-organized', 'keeps them neat', 'good case', 'easy access', 'tray', 'storage'],
        '负面-收纳不便': ['hard to get out', 'messy organization', 'case doesn\'t close'],
        },

    # ===== 10. 多样性与适配性 (恢复并优化) =====
        '多样性与适配性': {
        '正面-用途广泛': ['versatile', 'multi-purpose', 'all-in-one', 'many uses', 'works on many surfaces', 'good for everything'],
        '负面-用途单一': ['not versatile', 'only for paper', 'limited use'],
        '正面-可拓展性 (Collection can be expanded)': ['expandable collection', 'new colors available', 'can add to my collection', 'love the new sets', 'limited edition colors'],
        '负面-可拓展性差 (Poor expandability)': ['no new colors', 'collection is limited', 'wish they had more shades', 'no new sets released'],
        '正面-可补充性 (Can be replenished)': ['buy individually', 'open stock', 'refillable', 'can buy single pens', 'replacement available', 'love that I can replace'],
        '负面-可补充性差 (Poor replenishability)': ['can\'t buy single', 'not sold individually', 'wish they sold refills', 'no replacement nibs', 'have to buy a whole new set', 'forced to rebuy set'],
        },

    # ===== 11. 教育与启发 (恢复并优化) =====
        '教育与启发': {
        '正面-激发创意/乐趣': ['fun to use', 'inspiring', 'motivating', 'relaxing', 'joy', 'therapeutic', 'satisfying', 'makes me want to create', 'spark creativity'],
        '正面-适合初学者': ['beginner friendly', 'easy to start', 'good for beginners', 'great starter set'],
        '负面-有学习门槛': ['learning curve', 'not for beginners', 'hard to use', 'confusing'],
        '正面-有教学支持': ['good tutorial', 'helpful guide', 'great community'],
        '负面-无教学支持': ['no instructions', 'confusing guide']
        },

    # ===== 12. 特殊用途 =====
        '特殊用途': {
        '正面-专业级表现': ['professional grade', 'artist grade', 'pro grade', 'professional results', 'industry standard', 'lightfast', 'archival quality'],
        '负面-非专业级': ['not professional grade', 'hobby grade', 'student grade'],
        '正面-适用于特殊表面': ['works on fabric', 'good on glass', 'great on wood', 'permanent on plastic'],
        '负面-不适用于特殊表面': ['doesn\'t work on fabric', 'wipes off glass']
        },


        '性价比': {
        '正面-性价比高': ['price', 'value', 'deal', 'affordable', 'cheap', 'budget', 'good value', 'great deal', 'worth the money', 'great buy', 'reasonable price', 'cheaper than', 'alternative to'],
        '负面-价格昂贵': ['expensive', 'overpriced', 'not worth', 'pricey', 'costly', 'rip off', 'too much', 'waste of money']
        },

       '配套与服务': {
        '正面-提供色卡/好用': ['comes with a swatch card', 'includes a swatch card', 'love the swatch card',  'helpful swatch card', 'great for swatching', 'easy to swatch',
            'blank swatch card', 'pre-printed swatch card'],
        '负面-缺少色卡/不好用': [ 'no swatch card', 'wish it had a swatch card', 'doesn\'t come with a swatch card', 'had to make my own swatch card', 'swatch card is inaccurate',
            'swatch card is useless', 'colors on swatch card don\'t match' ]},

        '购买与服务体验': {'正面-开箱/展示': ['beautiful presentation', 'great unboxing experience', 'perfect for a gift', 'looks professional'],'负面-运输/损坏': ['arrived broken', 'leaking ink', 'damaged during shipping', 'box was crushed'],
        '正面-客服/售后': ['great customer service', 'seller was helpful', 'fast replacement', 'easy refund'],'负面-客服/售后': ['bad customer service', 'seller was unresponsive', 'missing items', 'wrong item sent']
    },
}

# 2. 定义专属“画像”，每个画像只包含需要“覆写”或“新增”的特殊规则。
PROFILE_OVERRIDES = {
    "默认基础画像": {}, # 这是一个空字典，选择它意味着只使用基础规则，不进行任何覆写。
    "霓虹笔专属画像": {
        '色彩表现': {
            # 对于霓虹笔，“太亮”是极致的赞美，因此新增一个正面评价。
            '正面-达到或超越期望的亮度': ['too bright', 'insanely bright', 'blindingly bright'],
            # 将基础规则中的“过饱和”负面评价清空，因为它不再适用。
            '负面-荧光/霓虹效果过饱和': [] 
        }
    },
    "香味笔专属画像": {
        '气味': {
            # 新增一个正面评价
            '正面-香味符合描述': ['smells good', 'great scent', 'smells like real fruit'],
            # 覆写基础规则，让负面评价更具体
            '负面-有异味': ['bad smell', 'chemical smell'], 
            # 新增一个负面评价
            '负面-没有香味': ['no smell', "can't smell anything", 'no scent'] 
        }
    }
}

# 3. 基础的用户分类规则
BASE_CLASSIFICATION_RULES = {
    "User_Role": {
                # 该分类已足够细致，保持 V6.5 版本
                '专业艺术工作者 (Professional Artist)': ['professional', 'pro artist', 'artist', 'illustrator', 'designer', 'comic artist', 'manga artist', 'architect', 'studio', 'commission', 'client work', 'freelance'],
                '学生 (Student)': ['student', 'school', 'college', 'university', 'art student', 'design student', 'class', 'notes', 'studying', 'assignment', 'project', 'textbook'],
                '教师 (Teacher)': ['teacher', 'educator', 'professor', 'art teacher', 'instructor', 'workshop', 'teaching', 'grading papers'],
                '父母 (Parent)': ['parent', 'mom', 'dad', 'mother', 'father', 'for my kids', 'for my son', 'for my daughter', 'family craft', 'homeschooling'],
                '手账爱好者 (Journaler/Planner)': ['journaler', 'planner', 'bullet journal', 'bujo', 'scrapbooker', 'diary', 'journaling', 'scrapbooking'],
                '业余艺术爱好者 (Hobbyist)': ['hobbyist', 'amateur artist', 'for fun', 'relaxing', 'as a hobby', 'passion project', 'in my spare time', 'self-taught'],
                '文化创意从业者 (Creative Professional)': ['creative professional', 'workshop host', 'cultural event', 'artisan', 'craft market', 'etsy seller', 'small business', 'content creator'],
                '特殊领域从业者 (Specialist)': ['special effects', 'sfx makeup', 'model maker', 'miniature painter', 'restorer', 'conservation', 'tattoo artist', 'animator'],
                '初学者 (Beginner)': ['beginner', 'starter', 'new to', 'learning', 'just starting', 'first set', 'noob', 'getting started', 'beginner friendly'],
                '商务/办公人士 (Business/Office Professional)': ['office', 'work', 'business', 'professional', 'presentation', 'meeting', 'notes', 'mind map', 'whiteboard', 'corporate', 'coworker', 'report', 'document', 'organization', 'organizing', 'at my desk'],
                '艺术疗愈/健康追求者 (Art Therapy/Wellness Seeker)': ['therapy', 'therapeutic', 'relax', 'relaxation', 'calming', 'mindfulness', 'anxiety', 'stress relief', 'zen', 'unwind', 'mental health', 'escape', 'self-care', 'peaceful', 'meditative'],
                '机构/批量采购者 (Institutional/Bulk Purchaser)': ['for my classroom', 'for the office', 'bulk order', 'school supplies', 'church group', 'community center', 'our team', 'stock up', 'office supply', 'large quantity', 'donation', 'for the class'],
            },
            "Gender": {'女性 (Female)': ['woman', 'women', 'girl', 'girls', 'she', 'her', 'hers', 'wife', 'mother', 'mom', 'daughter', 'girlfriend', 'female', 'sister', 'aunt', 'grandmother', 'niece', 'lady', 'ladies'],
                 '男性 (Male)': ['man', 'men', 'boy', 'boys', 'he', 'his', 'him', 'husband', 'father', 'dad', 'son',  'boyfriend', 'male', 'brother', 'uncle', 'grandfather', 'nephew', 'gentleman']
                       },
            "Age_Group":{ '儿童 (Child)': ['kid', 'kids', 'child', 'children', 'toddler', 'baby', 'preschooler', 'little one',  'for my son', 'for my daughter', 'grandson', 'granddaughter' ],'青少年 (Teenager)': ['teen', 'teenager', 'adolescent', 'youth', 'high school', 'college student', 'university student' ],
                         '老年人 (Senior)': [ 'senior', 'elderly', 'retired', 'grandparent', 'grandfather', 'grandmother', 'golden years' ]},

            "Usage": {
                #【V6.6 子类目细化】
                '绘画创作 (Art Creation)': ['art', 'drawing', 'illustration', 'manga', 'comic', 'landscape sketch', 'urban sketching', 'coloring book', 'artwork', 'painting', 'portrait', 'character design'],
                '设计工作 (Design Work)': ['design', 'architecture', 'industrial design', 'fashion design', 'concept art', 'floor plan', 'blueprint', 'storyboard', 'graphic design'],
                '教学与学习 (Teaching & Learning)': ['art class', 'craft class', 'workshop', 'tutorial', 'teaching', 'art school', 'student work', 'demonstration', 'learning to draw'],
                '手账装饰 (Journal & Planner Decoration)': ['journal', 'planner', 'bujo', 'diary', 'journaling', 'scrapbook', 'scrapbooking', 'decorating my planner'],
                '日常记录与组织 (Daily Organization)': ['calendar', 'labeling', 'organizing', 'note taking', 'annotating', 'study notes', 'meeting notes', 'color coding'],
                '卡片与礼品制作 (Card & Gift Making)': ['card making', 'greeting card', 'handmade card', 'gift tag', 'personal touch', 'decorating gifts'],
                '儿童涂鸦与早教 (Kids Activities)': ['kids', 'children', 'toddler', 'doodle', 'scribble', 'early learning', 'educational', 'kids craft', 'family fun'],
                'DIY与手工制作 (DIY & Crafts)': ['diy', 'craft', 'crafting', 'decorating', 'glass', 't-shirt', 'fabric', 'model painting', 'miniature painting', 'customizing', 'rock painting', 'mug decoration'],
                '户外与旅行创作 (Outdoor & Travel Art)': ['outdoor', 'en plein air', 'urban sketching', 'travel journal', 'traveling', 'on the go', 'field sketch'],
                '收藏与展示 (Collection & Display)': ['collection', 'collector', 'limited edition', 'collectible set', 'display'],
                '文化体验与活动 (Cultural Activities)': ['workshop', 'art event', 'cultural festival', 'live drawing', 'art therapy session', 'community art'],
                '心理疗愈 (Therapeutic Use)': ['relax', 'relaxation', 'stress relief', 'therapy', 'therapeutic', 'calming', 'mindfulness', 'emotional outlet', 'doodling', 'zen', 'wind down']
            },
            "Motivation": {
                #【V6.6 子类目细化】
                '专业需求-色彩表现': ['professional', 'artist grade', 'high quality pigment', 'lightfast', 'archival', 'color accuracy', 'blendability', 'vibrant colors'],
                '专业需求-性能耐用': ['pro grade', 'reliable', 'consistent flow', 'durable tip', 'long lasting', 'for work', 'serious tool'],
                '基础功能需求': ['basic', 'everyday use', 'daily use', 'for school', 'for notes', 'functional', 'practical', 'gets the job done', 'all i need'],
                '艺术兴趣驱动': ['hobby', 'passion', 'creativity', 'express myself', 'ideas', 'for fun', 'artistic', 'wanted to try', 'get back into art'],
                '情感表达': ['express feelings', 'handmade card', 'personal touch', 'gift for', 'decorate', 'scrapbook', 'memory keeping'],
                '品牌信任': ['brand', 'reputation', 'trusted brand', 'well-known', 'reliable', 'never fails', 'go-to brand', 'copic', 'tombow', 'stabilo', 'posca', 'winsor newton'],
                '性价比驱动': ['value', 'price', 'affordable', 'budget', 'deal', 'cheap', 'good price', 'cost effective', 'best bang for the buck', 'on sale'],
                '创新功能吸引': ['innovative', 'new feature', 'dual tip', 'refillable', 'replaceable nib', 'unique', 'special', 'interesting', 'different from others', 'new technology'],
                '外观设计吸引': ['design', 'aesthetic', 'beautiful', 'looks good', 'pretty colors', 'minimalist', 'stylish', 'the look of it', 'elegant'],
                '包装与开箱体验吸引': ['packaging', 'unboxing experience', 'giftable', 'nice box', 'presentation'],
                '社交驱动-口碑推荐': ['recommendation', 'recommended by', 'friend', 'family', 'teacher', 'word of mouth', 'told me to buy'],
                '社交驱动-媒体影响': ['social media', 'tiktok', 'instagram', 'youtube review', 'influencer', 'trending', 'hype', 'popular', 'everyone has it', 'pinterest'],
                '文化与身份认同': ['culture', 'themed set', 'limited edition', 'collaboration', 'artist series', 'Japanese', 'kawaii', 'collectible', 'part of my identity'],
                '便携性需求': ['convenient', 'portable', 'on the go', 'easy to carry', 'travel set', 'compact', 'all-in-one'],
                '多功能性需求': ['versatile', 'multi-purpose', 'many uses', 'for different things', 'one set for all', 'jack of all trades'],
                '礼品需求': ['gift', 'present', 'for someone', 'birthday', 'christmas', 'holiday', 'stocking stuffer', 'perfect gift'],
                '特殊场景需求': ['special purpose', 'outdoor', 'on glass', 'fabric marker', 'uv resistant', 'on black paper', 'for rocks', 'for wood'],
                '成就感与身份认同': ['achievement', 'feel like a pro', 'professional', 'identity', 'high-end', 'premium', 'top of the line', 'an investment', 'treat myself'],
                '激发创造力': ['inspiration', 'inspire', 'creativity', 'creative block', 'new ideas', 'get the juices flowing', 'unleash creativity'],
                '缓解压力与情绪调节': ['stress relief', 'relaxing', 'calming', 'therapy', 'therapeutic', 'mindfulness', 'escape', 'zone out', 'anxious', 'anxiety'],
                '满足好奇心': ['curiosity', 'try', 'try out', 'new', 'curious about', 'wanted to see', 'heard about', 'first impression'],
                '环保与可持续性': ['eco-friendly', 'sustainable', 'recycled', 'refillable', 'non-toxic', 'environment', 'less waste', 'conscientious'],
                '支持特定文化': ['local artist', 'local brand', 'cultural collaboration', 'support local', 'national pride'],
                '追随潮流': ['trend', 'trending', 'hype', 'popular', 'everyone has it', 'fashionable', 'in style', 'latest'],
                '效率驱动': ['efficient', 'efficiency', 'quick drying', 'fast', 'save time', 'work faster', 'streamline', 'slow drying'],
                '学习新技能': ['learn', 'learning', 'new skill', 'improve', 'get better', 'tutorial', 'starter kit', 'for beginners'],
                '提升现有技能': ['upgrade', 'next level', 'challenge myself', 'advanced techniques', 'better tool', 'step up my game']
            }
}

# --- 动态ASIN分类管理函数 ---
if 'category_mappings' not in st.session_state:
    st.session_state.category_mappings = []

def add_mapping():
    asin_input = st.session_state.new_asin
    category_input = st.session_state.new_category
    if asin_input and category_input:
        st.session_state.category_mappings.append({'asin': asin_input, 'category': category_input})
        st.session_state.new_asin = ""
        st.session_state.new_category = ""
    else:
        st.warning("ASIN和产品系列名称都不能为空！")

def delete_mapping(index_to_delete):
    st.session_state.category_mappings.pop(index_to_delete)

# --- 侧边栏：用户输入区域 ---
with st.sidebar:
    st.header("1. 上传文件")
    uploaded_file = st.file_uploader("请选择一个Excel文件", type=["xlsx"])

    st.header("2. 选择画像")
    # 让用户从我们定义的画像中选择一个
    selected_profile = st.selectbox("请选择最匹配您产品的画像", list(PROFILE_OVERRIDES.keys()))

    st.header("3. (选填) 添加用户分类")
    additional_roles_text = st.text_area("按JSON格式添加临时的用户角色", '{"新角色示例": ["关键词1", "关键词2"]}')

    with st.expander("步骤4: (重要) 自定义ASIN产品分类", expanded=True):
        st.markdown("###### 添加新的映射")
        col1, col2 = st.columns(2)
        col1.text_input("输入 ASIN", key="new_asin", placeholder="例如 B07C1BRS5N")
        col2.text_input("输入产品系列名称", key="new_category", placeholder="例如 柔色系列")
        st.button("添加映射", on_click=add_mapping, use_container_width=True)
        
        st.markdown("---")
        st.markdown("###### 已添加的映射")
        if not st.session_state.category_mappings:
            st.caption("尚未添加任何分类映射。")
        for i, mapping in enumerate(st.session_state.category_mappings):
            col1, col2, col3 = st.columns([3, 4, 1])
            col1.text(mapping['asin'])
            col2.text(mapping['category'])
            col3.button("❌", key=f"del_{i}", on_click=delete_mapping, args=(i,))

    st.markdown("---")
    analyze_button = st.button("开始生成报告", type="primary", use_container_width=True)

# --- 主界面：显示结果 ---
if analyze_button and uploaded_file is not None:
    file_buffer = io.BytesIO(uploaded_file.getvalue())
    
    with st.status('报告生成中，请稍候...', expanded=True) as status:
        try:
            # 1. 从session_state中构建最终的CATEGORY_MAPPING字典
            final_category_mapping = {item['asin'].lower(): item['category'] for item in st.session_state.category_mappings}
            
            # 2. 动态构建最终配置
            status.write("步骤 1/8: 正在构建分析配置...")
            final_config = {
                "input_filepath": file_buffer,
                "output_filepath": "processed_data.csv",
                "report_output_path": "final_report.html",
                "content_column": "Content", "rating_column": "Rating", "model_column": "Asin", "date_column": "Date",
                "keywords": [],
                "sentiment_bins": [-float('inf'), -0.05, 0.05, float('inf')],
                "sentiment_labels": ['Negative', 'Neutral', 'Positive'],
                "category_mapping": final_category_mapping,
                # 核心改动：将“基础”和“覆写”规则分别传入
                "base_keywords": BASE_FEATURE_KEYWORDS,
                "profiles": PROFILE_OVERRIDES,
                "classification_rules": BASE_CLASSIFICATION_RULES.copy(),
                "user_diagnostic_columns": ['User_Role', 'Gender', 'Age_Group']
            }

            try:
                if additional_roles_text and additional_roles_text.strip() != '{"新角色示例": ["关键词1", "关键词2"]}':
                    new_roles = json.loads(additional_roles_text)
                    final_config['classification_rules']['User_Role'].update(new_roles)
            except Exception:
                pass

            # 3. 初始化分析器并运行核心分析
            # ReviewAnalyzer的__init__方法会自动处理“基础+覆写”的合并逻辑
            status.write("步骤 2/8: 正在运行核心分析引擎...")
            analyzer = ReviewAnalyzer(config=final_config, product_type=selected_profile)
            processed_df = analyzer.run_analysis()

            if processed_df is None:
                raise ValueError("核心分析失败，未能生成DataFrame。请检查输入文件。")

            # 4. 执行所有分类
            status.write("步骤 3/8: 正在执行用户画像分类...")
            analyzer.classify_by_rules('User_Role', 'User_Role', '未明确')
            analyzer.classify_by_rules('Gender', 'Gender', '未知性别')
            analyzer.classify_by_rules('Age_Group', 'Age_Group', '成人')
            analyzer.classify_by_rules('Usage', 'Usage', '未明确')
            analyzer.classify_by_rules('Motivation', 'Motivation', '未明确')

            # 5. 生成时间维度
            status.write("步骤 4/8: 正在生成时间维度...")
            date_col = final_config['date_column']
            time_periods = {"_ALL_": "全部时间"}
            if date_col in processed_df.columns:
                processed_df[date_col] = pd.to_datetime(processed_df[date_col], errors='coerce')
                df_dated = processed_df.dropna(subset=[date_col]).copy()
                if not df_dated.empty:
                    df_dated['Year'] = df_dated[date_col].dt.year
                    df_dated['Quarter'] = df_dated[date_col].dt.to_period('Q').astype(str)
                    for year in sorted(df_dated['Year'].unique(), reverse=True):
                        time_periods[str(year)] = f"{year}年 全年"
                    for quarter in sorted(df_dated['Quarter'].unique(), reverse=True):
                        time_periods[quarter] = f"{quarter.replace('Q', '年 第')}季度"
                    processed_df = pd.merge(processed_df, df_dated[['Year', 'Quarter']], left_index=True, right_index=True, how='left')

            # 6. 按时间段循环执行深度诊断
            status.write("步骤 5/8: 正在执行深度诊断分析...")
            drill_down_reports_by_period = {}
            for period_key, period_label in time_periods.items():
                if period_key == "_ALL_": period_df = processed_df
                elif 'Q' in period_key: period_df = processed_df[processed_df['Quarter'] == period_key]
                else: period_df = processed_df[processed_df['Year'] == int(period_key)]
                if len(period_df) < 10: continue

                analyzer.df = period_df.copy()
                feature_reports = analyzer.run_comprehensive_feature_diagnostics()
                user_reports = analyzer.run_comprehensive_user_diagnostics()
                drill_down_reports_by_period[period_key] = feature_reports + user_reports
            analyzer.df = processed_df.copy()

            # 7. 宏观分析和准备最终数据包
            status.write("步骤 6/8: 正在准备仪表盘数据...")
            feature_report = analyzer.generate_feature_analysis_report()
            
            def format_crosstab_for_html(df: pd.DataFrame, index_name: str) -> Dict:
                df_reset = df.reset_index()
                for col in df_reset.columns:
                    if col != index_name: df_reset[col] = df_reset[col].map('{:.1f}%'.format)
                return {"headers": df_reset.columns.tolist(), "rows": df_reset.values.tolist()}
            
            role_preference_percentage = pd.crosstab(index=processed_df['User_Role'], columns=processed_df['Product_Category'], normalize='index') * 100
            gender_preference_percentage = pd.crosstab(index=processed_df['Gender'], columns=processed_df['Product_Category'], normalize='index') * 100
            age_group_preference_percentage = pd.crosstab(index=processed_df['Age_Group'], columns=processed_df['Product_Category'], normalize='index') * 100
            
            rating_counts = processed_df['Rating'].value_counts().sort_index()
            monthly_reviews = processed_df.set_index(date_col).resample('M').size() if date_col in processed_df.columns and not processed_df[date_col].isnull().all() else pd.Series()

            dashboard_data = {
                "totalReviews": len(processed_df),
                "avgRating": f"{processed_df['Rating'].mean():.2f}",
                "positiveRate": f"{(processed_df[processed_df['Rating'] >= 4].shape[0] / len(processed_df) * 100):.1f}%",
                "ratingDistribution": {"labels": [f"{i}星" for i in rating_counts.index], "data": rating_counts.values.tolist()},
                "reviewTrend": {"labels": [str(x.to_period('M')) for x in monthly_reviews.index], "data": monthly_reviews.values.tolist()} if not monthly_reviews.empty else {},
                "sentimentAnalysis": {"labels": processed_df['Sentiment_Category'].value_counts().index.tolist(), "data": processed_df['Sentiment_Category'].value_counts().values.tolist()},
                "userRoles": {"labels": processed_df['User_Role'].value_counts().index.tolist(), "data": processed_df['User_Role'].value_counts().values.tolist()},
                "genderDistribution": {"labels": processed_df['Gender'].value_counts().index.tolist(), "data": processed_df['Gender'].value_counts().values.tolist()},
                "ageGroupDistribution": {"labels": processed_df['Age_Group'].value_counts().index.tolist(), "data": processed_df['Age_Group'].value_counts().values.tolist()},
                "usageAnalysis": {"labels": processed_df['Usage'].value_counts().index.tolist(), "data": processed_df['Usage'].value_counts().values.tolist()},
                "purchaseMotivation": {"labels": processed_df['Motivation'].value_counts().index.tolist(), "data": processed_df['Motivation'].value_counts().values.tolist()},
                "rolePreferences": format_crosstab_for_html(role_preference_percentage, 'User_Role'),
                "genderPreferences": format_crosstab_for_html(gender_preference_percentage, 'Gender'),
                "ageGroupPreferences": format_crosstab_for_html(age_group_preference_percentage, 'Age_Group'),
                "featureSentimentStats": feature_report.get('feature_sentiment_stats', {}),
                "featureMentionRates": feature_report.get('rating_group_mention_rates', {}),
                "highRatingWordCloudData": [{"text": word, "size": count} for word, count in feature_report.get('word_frequencies', {}).get('high_rating_words', [])],
                "lowRatingWordCloudData": [{"text": word, "size": count} for word, count in feature_report.get('word_frequencies', {}).get('low_rating_words', [])],
                "drillDownTimePeriods": time_periods,
                "drillDownReports": drill_down_reports_by_period
            }

            # 8. 保存CSV并导出HTML报告
            status.write("步骤 7/8: 正在生成CSV数据文件...")
            analyzer.save_results()
            status.write("步骤 8/8: 正在生成HTML报告文件...")
            analyzer.export_to_html(dashboard_data)
            
            status.update(label="报告生成完毕！", state="complete", expanded=False)

        except Exception as e:
            st.error(f"在分析过程中发生严重错误: {e}")
            st.exception(e)
            status.update(label="分析失败", state="error")

    if 'final_config' in locals() and 'dashboard_data' in locals():
        st.success("🎉 分析流程已完成！现在您可以下载结果文件。")
        
        col1, col2 = st.columns(2)
        with col1:
            with open(final_config['report_output_path'], "rb") as file:
                st.download_button(
                    label="点击下载HTML报告",
                    data=file,
                    file_name=final_config['report_output_path'],
                    mime="text/html",
                    use_container_width=True,
                    type="primary"
                )
        with col2:
            with open(final_config['output_filepath'], "rb") as file:
                st.download_button(
                    label="点击下载CSV数据",
                    data=file,
                    file_name=final_config['output_filepath'],
                    mime="text/csv",
                    use_container_width=True
                )

elif analyze_button and uploaded_file is None:
    st.error("请先在左侧边栏上传一个Excel文件！")
else:
    st.info("请在主界面查看分析进度和下载最终报告。")
