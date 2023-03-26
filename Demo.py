import gradio as gr
from caption import captioning
import argparse
from cap_to_hashtag import cap2hashtag


parser = argparse.ArgumentParser()
parser.add_argument('--cap-dir', default='OFA-base', help='caption cap_model dir')
parser.add_argument('--cls-dir', default='', help='classification cap_model dir')
parser.add_argument('--folder-path', default='test-img/dir1', help='image dir')
parser.add_argument('--pred-num', default=5, help='predict how many classes')
parser.add_argument('--img-format', default='jpg', help='Image format')
parser.add_argument('--annot-path', default='test-img', help='annotation path')
arguments = parser.parse_args()


def hashtag_generation(*args, arugments=arguments):
    caption_list=captioning(*args, arguments=arguments)
    core, relative, impression = cap2hashtag(caption_list)
    # list
    return core, relative, impression
    

def copy(text_output=str, final_output=str):
    return final_output + ', ' +text_output


def main():

    with gr.Blocks() as demo:
        gr.Markdown(" <center><h1> Instagram Hashtag Generator </h1> </center>")
        with gr.Row():
            with gr.Column():
                bool_affluent_hashtags = gr.Checkbox(label = "Do you want more affluent recommentation using wordmap?")
                bool__hashtags = gr.Checkbox(label = "Do you want more hashtags for impression?")
                image_input1 = gr.Image(type='pil')
                image_input2 = gr.Image(type='pil')
                image_input3 = gr.Image(type='pil')
                image_input4 = gr.Image(type='pil')
                image_input5 = gr.Image(type='pil')
                input_bttn = gr.Button("Submit")
                #examples = gr.Examples(examples=['img/cheetah.jpg', 'img/elephang.jpg', 'img/giraffe.jpg', 'img/hippo.jpg', 'img/lion.jpg'])

            with gr.Column(): # output1 (core), output2 (related), output3 (most likeable), total
                
                gr.Markdown(" <center><h3> The most relative hashtags of your photos </h3> </center>")
                text_output1 = gr.Textbox(interactive=True, lines=4)
                acceptance_1 = gr.Button("Accept All")
                gr.Markdown(" <center><h3> More affluent recommendation results </h3> </center>")
                text_output2 = gr.Textbox(interactive=True, lines=4)
                acceptance_2 = gr.Button("Accept All")
                gr.Markdown(" <center><h3> Also these are for impressions </h3> </center>")
                text_output3 = gr.Textbox(interactive=True, lines=4)
                acceptance_3 = gr.Button("Accept All")
                final_output = gr.TextArea()
            #=========================================================================================================================
            input_bttn.click(hashtag_generation, inputs=[image_input1, image_input2, image_input3, image_input4, image_input5],\
                              outputs=[text_output1, text_output2, text_output3])
            #=========================================================================================================================
            #input_bttn.click(hashtag_generation, inputs=[image_input1, image_input2, image_input3, image_input4, image_input5], outputs=[text_output1, text_output2, text_output3])
            acceptance_1.click(copy, inputs=[text_output1, final_output], outputs=final_output)
            acceptance_2.click(copy, inputs=[text_output2, final_output], outputs=final_output)
            acceptance_3.click(copy, inputs=[text_output3, final_output], outputs=final_output)

    demo.launch(share=True)

if __name__=="__main__":
    main()