import argparse
import gradio as gr
from image_to_caption_dif import img2cap
from cls import img2cls
from cap_to_hashtag_bert import cap2hashtag
from image2text import vd_inference

parser = argparse.ArgumentParser()
parser.add_argument('--cap-dir', default='OFA-base',
                    help='caption cap_model dir')
parser.add_argument('--cls-dir', default='',
                    help='classification cap_model dir')
parser.add_argument('--folder-path', default='test-img/dir1', help='image dir')
parser.add_argument('--pred-num', default=5, help='predict how many classes')
parser.add_argument('--img-format', default='jpg', help='Image format')
parser.add_argument('--annot-path', default='test-img', help='annotation path')
arguments = parser.parse_args()

max_imageboxes = 10

vd_infer = vd_inference(which='v1.0', fp16=True)


def variable_outputs(k):
    k = int(k)

    # k개 visible, max_imagebox-k 개 invisible
    return [gr.Image(type='pil').update(visible=True)]*k + [gr.Image(type='pil').update(visible=False)]*(max_imageboxes-k)


def hashtag_generation(*args, model=vd_infer, arugments=arguments):
    caption_list = img2cap(*args, model=model, arguments=arguments) 
    core, relative, impression = cap2hashtag(caption_list)
    if args[-2] == False:  # bool_affluent_hashtags
        relative = []
    if args[-1] == False:  # bool_hashtags
        impression = []

    return str(core).lstrip('[').rstrip(']').replace('\'', '').replace(', ', '').replace(' #', '#').replace(' ', '_'), \
        str(relative).lstrip('[').rstrip(']').replace('\'', '').replace(', ', '').replace(' #', '#').replace(' ', '_'), \
        str(impression).lstrip('[').rstrip(']').replace(
            '\'', '').replace(', ', '').replace(' #', '#').replace(' ', '_')


def copy(text_output=str, final_output=str):
    return final_output + text_output


def main():

    with gr.Blocks() as demo:
        gr.Markdown(
            " <center><h1> Instagram Hashtag Generator </h1> </center>")
        with gr.Column():
            with gr.Row():
                bool_affluent_hashtags = gr.Checkbox(
                    label="Do you want more affluent recommentation using wordmap?")
                bool_hashtags = gr.Checkbox(
                    label="Do you want more hashtags for impression?")

                s = gr.Slider(1, max_imageboxes, value=max_imageboxes,
                              step=1, label="Your Input Image Number:")
                imageboxes = []
                # 처음에 10개
            with gr.Row():
                for i in range(max_imageboxes):
                    t = gr.Image(type='pil')
                    imageboxes.append(t)

                # Input 개수 바꾸기
                s.change(variable_outputs, s, imageboxes)
                #print("이미지 개수: ", float(s.value))
            with gr.Row():
                input_bttn = gr.Button("Submit").style(full_width=True)
                #examples = gr.Examples(examples=['img/cheetah.jpg', 'img/elephang.jpg', 'img/giraffe.jpg', 'img/hippo.jpg', 'img/lion.jpg'])

            with gr.Row():  # output1 (core), output2 (related), output3 (most likeable), total
                # ===============================================
                with gr.Column(scale=1, min_width=200):
                    gr.Markdown(
                        " <center><h5>The most relative hashtags of your photos </h5> </center>")
                with gr.Column(scale=10):
                    core = gr.Textbox(interactive=True,
                                      lines=4)
                with gr.Column(scale=1, min_width=100):
                    acceptance_1 = gr.Button("Accept All")
            with gr.Row():  # output1 (core), output2 (related), output3 (most likeable), total
                # ===============================================
                with gr.Column(scale=1, min_width=200):
                    gr.Markdown(
                        " <center><h5> More affluent recommendation results </h5> </center>")
                with gr.Column(scale=10):
                    relative = gr.Textbox(interactive=True,
                                          lines=4)
                with gr.Column(scale=1, min_width=100):
                    acceptance_2 = gr.Button("Accept All")
            with gr.Row():  # output1 (core), output2 (related), output3 (most likeable), total
                # ===============================================
                with gr.Column(scale=1, min_width=200):
                    gr.Markdown(
                        " <center><h5> Also these are for impressions </h5> </center>")
                with gr.Column(scale=10):
                    impression = gr.Textbox(interactive=True,
                                            lines=4)
                with gr.Column(scale=1, min_width=100):
                    acceptance_3 = gr.Button("Accept All")

            with gr.Row():
                final_output = gr.TextArea()

            # =========================================================================================================================
            # Input이 gradio component들의 List가 되어야 하는데, tuple은 gradio component가 아님.
            input_bttn.click(hashtag_generation, inputs=imageboxes + [bool_affluent_hashtags, bool_hashtags],
                             outputs=[core, relative, impression])
            # =========================================================================================================================
            #input_bttn.click(hashtag_generation, inputs=[image_input1, image_input2, image_input3, image_input4, image_input5], outputs=[text_output1, text_output2, text_output3])
            acceptance_1.click(
                copy, inputs=[core, final_output], outputs=final_output)
            acceptance_2.click(
                copy, inputs=[relative, final_output], outputs=final_output)
            acceptance_3.click(
                copy, inputs=[impression, final_output], outputs=final_output)

    demo.launch(share=True)


if __name__ == "__main__":
    main()
