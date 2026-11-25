# Test examples: crossCon, 3 seeds, LLM picks one candidate (7.2% voiced / 14.0% silent)

40 random test sentences (seed 7), each recorded once voiced and once silent. Whisper-normalised.
**bold** = wrong word, **+word** = inserted, ~~word~~ = missed (deleted). WER per sentence in brackets.
In this sample: 21/40 voiced and 15/40 silent sentences are perfect.


1. **Ref:** yeah i can understand that but at least we do not have to worry about things like mosquitoes and other bugs when it is cold out
   - Voiced (8%): yeah i can understand that but at least we do not have to worry about things like **this** and other **bags** when it is cold out
   - Silent (8%): yeah i can understand that but at least we do not have to worry about things like ~~mosquitoes~~ and other **bug** when it is cold out

2. **Ref:** wow that is impressive where do you get them all from
   - Voiced (9%): wow that is impressive where **did** you get them all from
   - Silent (9%): wow that is impressive where do you get them ~~all~~ from

3. **Ref:** i have been better i have been feeling a bit down lately and have been trying to regain my old self
   - Voiced (19%): i have been better i have been feeling a bit down lately and **+i** have been trying to **beat** my **own** **stuff**
   - Silent (14%): i have been better i have been feeling a bit down lately and **+i** have been trying to **get** my **own** self

4. **Ref:** i do not know i just am everything is so different at night the shadows on the ground look like monsters to me
   - Voiced (17%): i do not know i just am everything is so different at night the **houses** on the **+crowd** **to** look like **most** to me
   - Silent (17%): i do not know i just am everything is so different at night the **shell** on the **grand** look like **+a** **monster** to me

5. **Ref:** i know what you are thinking you are wondering how i could possibly be so perfect for you
   - Voiced (11%): i know what you **think** ~~thinking~~ you are wondering how i could possibly be so perfect for you
   - Silent (50%): i know what you **did** ~~thinking~~ you **went** **and** **apt** **is** ~~could~~ ~~possibly~~ ~~be~~ so perfect for you

6. **Ref:** why what is wrong with what you are wearing
   - Voiced (0%): why what is wrong with what you are wearing
   - Silent (0%): why what is wrong with what you are wearing

7. **Ref:** yeah it was actually really beautiful the leaves are changing color and the scenery was just stunning
   - Voiced (12%): yeah it was actually really beautiful the leaves ~~are~~ changing color and the scenery was just **studying**
   - Silent (18%): yeah it was actually really beautiful the leaves are **+and** **the** color and the scenery was just **studying**

8. **Ref:** hi pamela how are you today
   - Voiced (0%): hi pamela how are you today
   - Silent (0%): hi pamela how are you today

9. **Ref:** i am so sorry to hear that i was not involved in any hit and run i swear
   - Voiced (28%): i am so sorry to hear that i was not **of** ~~in~~ any **one** ~~and~~ ~~run~~ i swear
   - Silent (28%): i am so sorry to hear that i was not **falling** in **that** **one** ~~and~~ ~~run~~ i swear

10. **Ref:** yeah things are good i am just keeping busy and trying to stay out of trouble
   - Voiced (0%): yeah things are good i am just keeping busy and trying to stay out of trouble
   - Silent (6%): **yes** things are good i am just keeping busy and trying to stay out of trouble

11. **Ref:** yeah i really appreciate it it is just been so tough lately you know trying to keep up with school and work and everything else
   - Voiced (0%): yeah i really appreciate it it is just been so tough lately you know trying to keep up with school and work and everything else
   - Silent (0%): yeah i really appreciate it it is just been so tough lately you know trying to keep up with school and work and everything else

12. **Ref:** it is just my dad has a drinking problem and i want to help him get better but i do not know how
   - Voiced (0%): it is just my dad has a drinking problem and i want to help him get better but i do not know how
   - Silent (13%): it is just my dad has a drinking problem and i want to help **you** get better but i ~~do~~ ~~not~~ know how

13. **Ref:** how am i supposed to find them
   - Voiced (0%): how am i supposed to find them
   - Silent (0%): how am i supposed to find them

14. **Ref:** cheryl that is not how it works titles do not automatically make people respect you or listen to you you still have to earn that
   - Voiced (28%): cheryl that is not how it works **+i** **+know** **you** do not **+want** **+a** **medically** make people respect you or listen to you you still have to **do** that
   - Silent (44%): cheryl that is not how it works **i** do not **+know** **not** **to** **medically** **my** **people** **and** **expect** to you you **will** have to **learn** that

15. **Ref:** that is great honey i am glad you are feeling better
   - Voiced (0%): that is great honey i am glad you are feeling better
   - Silent (0%): that is great honey i am glad you are feeling better

16. **Ref:** yes it was much more exciting than being a normal person
   - Voiced (0%): yes it was much more exciting than being a normal person
   - Silent (9%): **yeah** it was much more exciting than being a normal person

17. **Ref:** just watching tv nothing too exciting
   - Voiced (0%): just watching tv nothing too exciting
   - Silent (0%): just watching tv nothing too exciting

18. **Ref:** hey i am here for you
   - Voiced (0%): hey i am here for you
   - Silent (0%): hey i am here for you

19. **Ref:** i do not know i just feel like i could have done better i worked so hard and you just barely beat me
   - Voiced (0%): i do not know i just feel like i could have done better i worked so hard and you just barely beat me
   - Silent (4%): i do not know i just feel like i could have done better i worked so hard and you just ~~barely~~ beat me

20. **Ref:** i just feel so scared i do not want to leave this room it is safe here
   - Voiced (0%): i just feel so scared i do not want to leave this room it is safe here
   - Silent (0%): i just feel so scared i do not want to leave this room it is safe here

21. **Ref:** 0 thank you ethan it looks great
   - Voiced (29%): 0 thank you **+it** **then** it looks great
   - Silent (14%): 0 thank you ~~ethan~~ it looks great

22. **Ref:** thank you so much for the raise and bonus i really appreciate it it means a lot to me
   - Voiced (5%): thank you so much for the raise and **posts** i really appreciate it it means a lot to me
   - Silent (11%): thank you so much for the raise ~~and~~ ~~bonus~~ i really appreciate it it means a lot to me

23. **Ref:** i was hoping you could help me become more virtuous
   - Voiced (10%): i was hoping you could help me become more **jokes**
   - Silent (20%): i was hoping you could help me become more **+to** **you**

24. **Ref:** i do not think so i just want to make sure everything is perfect before i start playing
   - Voiced (0%): i do not think so i just want to make sure everything is perfect before i start playing
   - Silent (0%): i do not think so i just want to make sure everything is perfect before i start playing

25. **Ref:** it is definitely one of the best i have been to the company is even better
   - Voiced (6%): it is definitely one of the best ~~i~~ have been to the company is even better
   - Silent (19%): it is definitely one of the best ~~i~~ have been to ~~the~~ company ~~is~~ even better

26. **Ref:** that is so cool i wish i could have gone with you
   - Voiced (0%): that is so cool i wish i could have gone with you
   - Silent (0%): that is so cool i wish i could have gone with you

27. **Ref:** tyler you are scaring me let me in please
   - Voiced (22%): **hi** you are **scary** me let me in please
   - Silent (22%): ~~tyler~~ you are **carrying** me let me in please

28. **Ref:** i am already halfway through and i can not put it down the characters are so well developed and the plot is really interesting
   - Voiced (17%): i am already **with** **school** and i can not **break** it down the characters are so well developed and the **book** is really interesting
   - Silent (25%): i am already **in** **school** and i can not put it down the characters ~~are~~ so well **the** **fall** the **body** is really interesting

29. **Ref:** yeah they can get pretty big around here
   - Voiced (0%): yeah they can get pretty big around here
   - Silent (0%): yeah they can get pretty big around here

30. **Ref:** you look great in that dress it really brings out your best features
   - Voiced (8%): you look great ~~in~~ that dress it really brings out your best features
   - Silent (15%): you look great ~~in~~ that **rice** it really brings out your best features

31. **Ref:** good i love you son
   - Voiced (0%): good i love you son
   - Silent (0%): good i love you son

32. **Ref:** okay that sounds good but i have never played before so you will have to go 1st
   - Voiced (0%): okay that sounds good but i have never played before so you will have to go 1st
   - Silent (12%): okay that sounds good but i have never played before ~~so~~ you ~~will~~ have to go 1st

33. **Ref:** well i am glad you were able to fix it
   - Voiced (0%): well i am glad you were able to fix it
   - Silent (0%): well i am glad you were able to fix it

34. **Ref:** i can not believe you are gone amanda i have looked everywhere for you and i just can not find you i miss you so much already
   - Voiced (4%): i can not believe you are gone **ana** i have looked everywhere for you and i just can not find you i miss you so much already
   - Silent (4%): i can not believe you are gone **man** i have looked everywhere for you and i just can not find you i miss you so much already

35. **Ref:** yes i do just give her a chance i think you will be surprised how wonderful she is
   - Voiced (17%): yes i do just give **it** a chance i think you will be surprised how wonderful ~~she~~ ~~is~~
   - Silent (17%): yes i do just give her a chance i think you will be surprised **and** wonderful ~~she~~ ~~is~~

36. **Ref:** hey i noticed that you looked pretty down lately what is going on
   - Voiced (0%): hey i noticed that you looked pretty down lately what is going on
   - Silent (0%): hey i noticed that you looked pretty down lately what is going on

37. **Ref:** hey robbie i saw you over here and wanted to buy you a drink
   - Voiced (7%): hey robbie i saw you over here and **+i** wanted to buy you a drink
   - Silent (14%): hey robbie i saw you over **there** and **+i** wanted to buy you a drink

38. **Ref:** yeah those are great games i am a big fan of call of duty myself do you guys have any other hobbies
   - Voiced (5%): yeah those are great games i am a big fan of call of duty myself do you guys have any other **options**
   - Silent (14%): yeah those are great games i am a big fan of **all** of **one** myself do you guys have any other **options**

39. **Ref:** that is good you deserve a break after all that hard work
   - Voiced (0%): that is good you deserve a break after all that hard work
   - Silent (0%): that is good you deserve a break after all that hard work

40. **Ref:** sounds good let me know what you find out
   - Voiced (0%): sounds good let me know what you find out
   - Silent (22%): **+that** **is** good let me know what you find out

