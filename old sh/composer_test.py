from PIL import Image
from composer import Composer

bg = Image.open("back.png")
fig = Image.open("comrade_№1.png")

#bg.show()
#fig.show()

comp = Composer(interactive=True, verbose=False)

a = comp.compose(
    background=bg,
    mode="full",
    figure=fig,
    position=(150, 320),
    scale=1.0)

a.collage.show()
#a.mask.show()

a.collage.save("collage.png")