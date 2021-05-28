add wave -position insertpoint  \
sim/:tb_bode_plot:A \
sim/:tb_bode_plot:initdone \
sim/:tb_bode_plot:clock \
sim/:tb_bode_plot:Z \

run -all
