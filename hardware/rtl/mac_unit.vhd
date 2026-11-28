
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

entity mac_unit is

    port (
        clk       : in  std_logic;
        rst       : in  std_logic;

        clear_acc : in  std_logic;
        load_bias : in  std_logic;

        bias_in   : in  signed(31 downto 0);

        mac_en    : in  std_logic;

        pixel_in  : in  unsigned(7 downto 0);
        weight_in : in  signed(7 downto 0);

        acc_out   : out signed(31 downto 0)
    );

end entity mac_unit;


architecture RTL of mac_unit is

    signal acc_reg : signed(31 downto 0);

begin

    process(clk)

        variable pixel_signed : signed(8 downto 0);
        variable product      : signed(16 downto 0);

    begin

        if rising_edge(clk) then

            if rst = '1' then

                acc_reg <= (others => '0');

            elsif clear_acc = '1' then

                acc_reg <= (others => '0');

            elsif load_bias = '1' then

                acc_reg <= bias_in;

            elsif mac_en = '1' then

                -- Convert unsigned pixel to positive signed value
                pixel_signed := signed('0' & std_logic_vector(pixel_in));

                -- Multiplication
                product := pixel_signed * weight_in;

                -- Accumulate product
                acc_reg <= acc_reg + resize(product, 32);

            end if;

        end if;

    end process;


    -- Connect internal accumulator to output
    acc_out <= acc_reg;

end architecture RTL;
