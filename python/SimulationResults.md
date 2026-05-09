## Tests with joker locking

| Strategy function                                       | # of wins |
|---------------------------------------------------------|-----------|
| maximize_value_always_saves_joker_and_joker_substitutes | 2628      |
| max_tiles_every_time                                    | 2446      |
| maximize_value_joker_value_one                          | 2428      |
| maximize_value_always_saves_joker                       | 2383      |

Total runtime: 6541 seconds, Empty bag games: 115

Conclusion: `maximize_value_always_saves_joker_and_joker_substitutes` wins by a statistically significant but small
margin, expected given more sophisticated strategy
___

| Strategy function                                           | # of wins |
|-------------------------------------------------------------|-----------|
| hoarder                                                     | 70        |
| 3 * maximize_value_always_saves_joker_and_joker_substitutes | 21        |

Total runtime: 93 seconds, Empty bag games: 9

Conclusion: hoarding is obviously the best strategy for a bot, or maybe even a human, but it is not fun to play.
___

| Strategy function                                                    | # of wins |
|----------------------------------------------------------------------|-----------|
| 2 * minimum_non_zero_placed_always_saves_joker_and_joker_substitutes | 582       |
| 2 * maximize_value_always_saves_joker_and_joker_substitutes          | 410       |

Total runtime: 997 seconds, Empty bag games: 8

Conclusion: `minimum_non_zero_placed_always_saves_joker_and_joker_substitutes` is the best non-hoarder strategy. Though
by trying to place the minimum tiles necessary, it is sort of hoarding. Despite that it is a much more "normal" strategy
that humans use, and it doesn't feel like BS. Another way of thinking about it is that you are trying to minimise the
number of times you need to pick up from the bag (though you could probably do that more optimally by looking ahead).
___

| Strategy function                                                    | # of wins |
|----------------------------------------------------------------------|-----------|
| hoarder                                                              | 76        |
| 3 * minimum_non_zero_placed_always_saves_joker_and_joker_substitutes | 13        |

Total runtime: 138 seconds, Empty bag games: 11

Conclusion: hoarding still crushes, probably by more than the value maximiser, but nobody wants to play with hoarder any
more.