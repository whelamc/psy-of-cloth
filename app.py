import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes
import colorsys

# 使用fasterRCNN 进行目标检测，使用预训练的权重
weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
categories = weights.meta["categories"]
img_preprocess = weights.transforms()


# 临时存储数据并引用模型
@st.cache_resource
def load_model():
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9)
    model.eval()
    return model

model = load_model()

# 实现目标检测
def make_prediction(img):
    img_processed = img_preprocess(img)
    prediction = model(img_processed.unsqueeze(0))
    prediction = prediction[0]
    prediction["labels"] = [categories[label] for label in prediction["labels"]]
    # 删除多余检测对象
    array = []
    for i in range(len(prediction["labels"])):
        if prediction["labels"][i] != "person":
            array.append(i)
    if len(array) != len(prediction["labels"]):
        boxes =  prediction["boxes"].detach().numpy().tolist()
        scores = prediction["scores"].detach().numpy().tolist()
        array.reverse()
        for x in range(len(array)):
            del boxes[int(array[x])]
            del scores[int(array[x])]
        prediction["boxes"] = torch.Tensor(boxes)
        prediction["scores"] = torch.Tensor(scores)
        prediction["labels"] = []
        for y in range(len(boxes)):
            prediction["labels"].append("person")
    return prediction

# 使用框框
def create_image_with_bboxes(img, prediction):
    img_tensor = torch.tensor(img)
    img_with_bboxes = draw_bounding_boxes(img_tensor, boxes=prediction["boxes"], labels=prediction["labels"],
                                          colors="red", width=1)
    img_with_bboxes_up = img_with_bboxes.detach().numpy().transpose(1, 2, 0)

    return img_with_bboxes_up

# 裁剪目标
def image_with_corp(img,prediction):
    new_prediction = prediction["boxes"].detach().numpy().tolist()[0]
    x1 = new_prediction[0]
    y1 = new_prediction[1]
    x2 = new_prediction[2]
    y2 = new_prediction[3]
    result = img.crop((x1,y1,x2,y2))
    return result

# 计算主要颜色
def get_dominant_color(img, palette_size=16):
    img = img.copy()
    img.thumbnail((100,100))
    paletted = img.convert("P",palette=Image.ADAPTIVE,colors=palette_size)
    palette = paletted.getpalette()
    color_counts = sorted(paletted.getcolors(),reverse=True)
    palette_index =color_counts[0][1]
    dominant_color = palette[palette_index*3:palette_index*3+3]
    return dominant_color

# 判断颜色存在的色系
def get_color_category(color):
    hsv_color = colorsys.rgb_to_hsv(color[0]/250, color[1]/250, color[2]/250)
    hue = hsv_color[0]
    saturation = hsv_color[1]
    value = hsv_color[2]

    if value < 0.2:
        return "黑色"
    elif value > 0.8 and saturation < 0.2:
        return "白色"
    elif saturation < 0.2:
        return "灰色"
    elif 0 <= hue <= 30 or 330 < hue <= 360:
        return "红色"
    elif 30 < hue <= 90:
        return "黄色"
    elif 90 < hue <= 150:
        return "绿色"
    elif 150 < hue <= 210:
        return "青色"
    elif 210 < hue <= 270:
        return "蓝色"
    elif 270 < hue <= 330:
        return "紫色"
    else:
        return "未知"


# 判断返回解说
def get_back_explain(color):
    if color == "蓝色":
        return "蓝色让人感到平静，象徵著智慧与理性，因此被心理学家认定为最适合穿去面试的颜色，这也是为什麽多数商务人士都喜欢穿蓝色套装或衬衫。喜欢穿蓝色的人往往理智且冷静，容易接纳并包容周边的人事物。而喜欢将蓝色与对比色搭配著穿的人，通常都有著天马行空般的想像力，创造力非常高。"

    if color == "红色":
        return "红色总是最吸引人目光，却又最难驾驭，爱穿红色的人，大多具有冒险精神，勇于接受挑战、拥有开阔的心胸接纳新鲜事物，这类型的人通常也擅长于处理人际关係。另外，不少人喜欢将红黑色穿衣搭配，象徵著慾望和野心，但往往因为有太多慾求，较难感到知足。"

    if color == "绿色":
        return "绿色让人联想到大自然，喜欢绿色的人通常处于一个稳定的生活环境裡，拥有一颗温暖的心，会主动关心并照顾朋友，态度积极、社交能力强，与多数的人都能和谐相处，不过看似落落大方，实际上是压抑自我慾望，较难走进内心，有时候也会过于安于现状。"

    if color == "黄色":
        return "鲜豔的黄色给人活泼开朗的印象，就像小小兵一般拥有天真烂漫的内心，对生活富有热情、拥有非凡的创意力，不过有时候却也因为这份天真，常被身边的人误认为幼稚。"

    if color == "紫色":
        return "以前，紫色被认为是皇室贵族的专属色，象徵著尊贵、奢华与财富，因此喜欢穿紫色的人，由内而外散发著一股高贵气息。这类型的人通常喜欢幻想、情感丰沛且浪漫，对生活充满激情，在艺术方面很有天分，不过心思较为敏感，相对来讲个性较难以捉摸，想要真正的了解他们，哇那还真是有点挑战呢！"

    if color == "白色":
        return "白色穿搭乾淨俐落，给人一种纯真无邪的印象。喜欢穿白色衣服的人是典型的完美主义者，个性乐观、追求完美与平衡，有时候无法容忍他人缺陷，潜意识裡其实很渴望引起别人的注意和关心。"

    if color == "黑色":
        return "喜欢黑色的人较不善于表达情感，他们通常压抑且不太吐露自己的感受，常常被不熟的人误认为冷漠高傲、不好亲近，但这其实这并非本性，你们也渴望被关注、被爱。穿全黑的人通常有点神经质，平时看似情绪没有波动，但是当看到有兴趣的事物就会特别亢奋，十分反差。"

    if color == "灰色":
        return "灰色被认为是中性和不偏倚的颜色，因此穿衣灰色可以传达出中立和不显眼的感觉。它可能适用于那些希望保持低调、不引人注目或不想过分突出的人。"

    if color == "未知":
        return "你很神秘"

# Dashboard
st.title("穿衣颜色心理")  # 设置标题
upload = st.file_uploader(label="上传图片", type=["png", "jpg", "jpeg"])  # 上传文件
if upload:
    img = Image.open(upload)
    prediction = make_prediction(img)
    img_with_bbox = create_image_with_bboxes(np.array(img).transpose(2, 0, 1), prediction)

    #作图
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)
    plt.imshow(img_with_bbox)
    plt.xticks([], [])
    plt.yticks([], [])
    ax.spines[["top", "bottom", "right", "left"]].set_visible(False)
    st.pyplot(fig, use_container_width=True)  # 绘图，显示matplotlib创建的图

    #概率展示
    if "person" in prediction["labels"] :
        img_with_corp = image_with_corp(img,prediction)
        dominant_color = get_dominant_color(img_with_corp)
        color_category = get_color_category(dominant_color)
        explanation = get_back_explain(color_category)
        st.header("解释",divider="rainbow")
        st.markdown(explanation)
        # st.write(prediction)
    else :
        st.header("没有检测到人",divider="rainbow")

st.markdown(''':rainbow[仓库: https://github.com/whelamc/psy-of-cloth]''')
