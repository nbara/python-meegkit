from unittest import TestCase, main, skip


class TestParseRaw(TestCase):

    def test_parse_raw_init(self):
        expected_board_type = k.BOARD_DAISY
        expected_gains = [24, 24, 24, 24, 24, 24, 24, 24]
        expected_log = True
        expected_micro_volts = True
        expected_scaled_output = False

        parser = ParseRaw(board_type=expected_board_type,
                          gains=expected_gains,
                          log=expected_log,
                          micro_volts=expected_micro_volts,
                          scaled_output=expected_scaled_output)

        self.assertEqual(parser.board_type, expected_board_type)
        self.assertEqual(parser.scaled_output, expected_scaled_output)
        self.assertEqual(parser.log, expected_log)


if __name__ == '__main__':
    main()
