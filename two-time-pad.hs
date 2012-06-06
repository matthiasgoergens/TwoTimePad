{-# Language NoMonomorphismRestriction #-}
import System.IO
import Data.List
import Data.Maybe
import Data.Ord
import Data.Char
import Data.Function
import qualified Data.Set as S
import qualified Data.Map as M
import qualified Data.IntMap as Mi
import Control.Monad
import System.Random
import Control.Arrow

{-
You have intercepted two encrypted messages, ciphertext-1 and ciphertext-2, encoded in the following 46-character alphabet:

[space]ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.?,-:;'()

The messages were encrypted with a one-time pad - a sequence of randomly drawn characters of the same length and alphabet as the plaintexts. The encryption algorithm is to add the code of a plaintext character with the code of the corresponding pad character, modulo 46. To illustrate, if the plaintext is "ITA SOFTWARE" and the pad is "9KS;UENFN068" then the ciphertext is ")4T;-TTZ.1E-".

plaintext: ITA SOFTWARE = 9 20 1 0 19 15 6 20 23 1 18 5
pad: 9KS;UENFN068 = 36 11 19 42 21 5 14 6 14 27 33 35
ciphertext: )4T;-TTZ.1E- = 45 31 20 42 40 20 20 26 37 28 5 40

However, the sender made the critical mistake of using the same one-time pad for both messages, compromising their security. Write a program that takes the two ciphertexts and produces a guess at the two plaintexts. It should produce two plaintext messages of the same length as the ciphertexts, hopefully containing many of the correct words in the correct positions. It is unlikely your program will get all of the words correct, since there is not enough information in the ciphertexts to uniquely determine the plaintexts. Strive for a quality of decryption sufficient for a human reader--perhaps with the aid of a web search--to identify the original texts, which happen to be excerpts from classic works of English literature.

Your program may use English text or word lists or dictionaries of your choice to train on, for example to gather tables of letter or word frequencies, but you should not rely on any portion of the messages being in your training material.

This puzzle was created September '04 and retired December '06.
-}

plaintext = map Alpha "ITA SOFTWARE"
pad =       map Alpha "9KS;UENFN068"
cipher =    map Alpha ")4T;-TTZ.1E-"

t = "ciphertext-2.txt"
-- t = "quantum.txt"


ti :: IO String
ti = liftM fil $ withFile t ReadMode (\h -> do
                               l <- hGetContents h
                               length l `seq` return l)

alpha = map toLower " ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.?,-:;'()"

len :: Num a => a
len = fromIntegral $ length alpha -- 46

from :: Mi.IntMap Int
from = Mi.fromList $ zip (map ord alpha) [0..]
to :: Mi.IntMap Alpha
to = Mi.fromList $ zip [0..] (map Alpha alpha)

instance Show Alpha where
    show (Alpha a) = "Alpha '" ++ a : "'"
    showList a s = ("Alpha " ++ show (map unAlpha a)) ++ s
      where unAlpha (Alpha a) = a

newtype Alpha = Alpha Char
instance Enum Alpha where
    fromEnum (Alpha c) = Mi.findWithDefault (error $ "Char out of range: " ++ show c) (ord . toLower $ c) from
    toEnum i = Mi.findWithDefault (error $ "Int out of Range: " ++ show i) i to

instance Num Alpha where
    a + b = toEnum $ (fromEnum a + fromEnum b) `mod` len
    a - b = toEnum $ (fromEnum a - fromEnum b) `mod` len

fil = filter isOK . map toLower . map toSpace
isOK x = S.member x (S.fromList $ alpha)
toSpace x | elem x " \t\n\r" = ' '
          | otherwise = x

stail [] = []
stail (_:xs) = xs
heads [] = Nothing
heads x = Just $ head x

trans :: String -> [String]
trans = sort . tails

-- printT = print . mapMaybe heads
printT = mapM print

-- 1/l * (sum_x x * log x) - log l

lb = logBase len

entropy n s = (\xs -> - (xs / groupLen - lb groupLen)) . sum . map ((\x -> x * lb x) . fromIntegral . length) $
              groups
    where l = fromIntegral $ length s
          groups = sortBy (flip $ comparing length) . group . map (take n) $ s
          groupLen = fromIntegral . sum . map length $ groups


geneticAlgo popCap initPop eval mutate crossover = do
    pop <- replicateM popCap initPop
    sortBy (comparing fst) $ map (eval &&& id) pop
    -- quadratic selection?
    replicateM popCap $ do select popCap

main = do
    -- print (zipWith (+) plaintext pad)
    -- print cipher
    -- print plaintext
    -- print (zipWith (-) cipher pad)
    -- print (zipWith (-) (zipWith (+) plaintext pad) pad)
    t <- liftM trans ti
    -- mapM print . map ((fst.head)&&&(length &&& (map snd))). groupBy ((==) `on` fst) . map (length &&& head) $
    print (entropy 3 t)
    -- printT =<< liftM trans ti
