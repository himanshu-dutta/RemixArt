# RemixArt

Regardless of whether it's a book cover, album art or only template for a simple project, we are continually searching for designs over the web. Indeed, even with some prefix in our mind,
we generally don't discover what we need. Our project aims at developing a model that takes inputs in the form of text, images and even audio and attempts to produce a picture or work of
art, maybe, as photorealistic as could be expected under the circumstances. To illustrate, we work with a dataset of songs, album covers, artist images, and song lyrics to generate a
close-to-real artwork. The idea can then be put to use in various domains also, where a lot of information in various formats are available. We use GANs with alterations made to
incorporate inputs from three unique channels, and with that, we train it to learn embedding based on every one of the three distinct channels.

_Adding the notes and changes related to the project, to keep track of it._

_Generating Album Art using ~3~ 2 channels of input, Audio, ~Images~ and/or Text._

# Himanshu's:

- [x] Model Selection and Learning to Apply It
- [x] Pytorch
- [x] Applying the model in Pytorch
- [x] Figuring Out How to Actually Transfer the Workflow to GCloud

# Archita's:

- [x] Data Scrapping and Collection
- [x] Deciding on the Data Source
- [ ] Storage and Retrival for efficinet processing, locally or over cloud buckets.
- [ ] ~Choice of Databse that would work well with the project.~

# Citation:

We leveraged the architecture of Stack GAN model in pytorch, with updates to the recent version of it, made fair share of modifications in terms of both, the procedure the original model followed along with the changes made to the conditional augmentation technique as well as embedding representation, with a vanilla model consisting of one dense layer and relu unit.

```
@inproceedings{han2017stackgan,
Author = {Han Zhang and Tao Xu and Hongsheng Li and Shaoting Zhang and Xiaogang Wang and Xiaolei Huang and Dimitris Metaxas},
Title = {StackGAN: Text to Photo-realistic Image Synthesis with Stacked Generative Adversarial Networks},
Year = {2017},
booktitle = {{ICCV}},
}
```
