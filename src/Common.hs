{-# LANGUAGE TupleSections, GeneralizedNewtypeDeriving #-}
module Common where
import Prelude hiding (length, sequence, mapM)
import Control.Monad (join, msum)
import Data.Char
import Data.Maybe
import Data.Int
import Data.Vector hiding (map, (++))
import Control.Arrow
import Control.Applicative
import Data.List (sort, groupBy)
import Data.Either
import Data.Function

import qualified Data.List as DL

import Data.Traversable as T

newtype C46 = C46 { unC46 :: Int8 }
            deriving (Ord, Eq, Show, Integral, Real, Enum)

plaintext = toLower <$> "ITA SOFTWARE"
pad = toLower <$>      "9KS;UENFN068"
cipher = toLower <$>    ")4T;-TTZ.1E-"


-- assert: test == True
test = Just cipher == (traverse join . (T.mapM.fmap) (hush . toChr_) $ DL.zipWith (+) <$> (T.mapM frChr_ plaintext ) <*> (T.mapM frChr_ pad ))

instance Num C46 where
  (+) (C46 a) (C46 b) = C46 $ (a + b) `mod` 46
  (-) (C46 a) (C46 b) = C46 $ (a - b) `mod` 46
  fromInteger = C46 . fromIntegral
  abs = C46 . abs . unC46
  signum = C46 . signum . unC46

alpha :: String
alpha = " abcdefghijklmnopqrstuvwxyz0123456789.?,-:;'()";
-- alpha = map toLower " ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.?,-:;'()"

toChr_ :: C46 -> Either String Char
toChr_ = fmap (chr . fromIntegral) . toChr

toChr :: C46 -> Either String Int8
toChr x | x < 0 = Left $ "toChr: too small " ++ show x ++ " < 0"
        | fromIntegral x >= length toChr' = Left $ "toChr: too larg " ++ show x ++ " > " ++ show (length toChr')
        | otherwise = Right $ toChr' ! fromIntegral x

toChr' :: Vector Int8
toChr' = fromList . fmap (fromIntegral . ord) $ alpha

frChr_ :: Char -> Maybe C46
frChr_ = frChr . fromIntegral . ord

frChr :: Int8 -> Maybe C46
frChr c = fmap C46 . DL.lookup c $ DL.zip (fromIntegral . ord <$> alpha) [0..]

hush :: Either a a1 -> Maybe a1
hush = either (const Nothing) Just

-- instance Traversable ((,) a) where
--  sequence t = undefined

{-
-- frChr' :: Vector (Maybe Int8)
--frChr' :: [[(Int8, Maybe Int8)]]
frChr' = -- fromList .
--  fmap snd .
  -- fmap msum .
  (fmap.fmap) T.sequence .
         groupBy ((==) `on` fst) .
         sort $
         l1 ++ l2
  where all :: [Int8]
        all = [0..maxBound]
        l1 :: [(Int8, Maybe Int8)]
        l1 = rights $ fmap (inv . (toChr &&& Just)) all
        inv (x, y) = fmap (,y) x
        l2 :: [(Int8, Maybe Int8)]               
        l2 = fmap (, Nothing) all
-}

{-

const char* const alphabet = " abcdefghijklmnopqrstuvwxyz0123456789.?,-:;'()";

void buildTables () {
  toTable = calloc (256, sizeof(char));
  fromTable = malloc (alpha_size * sizeof(char));
  for (int i = 0; i < 256; i++) {
    fromTable[i] = alpha_size; }
  const char* alpha = alphabet; int i=0;
  for (; *alpha; i++, alpha++) {
    toTable[(int) *alpha] = i;
    fromTable[i] = *alpha; }
  printf ("Done building.\n"); }

-}