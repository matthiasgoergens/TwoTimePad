{-# LANGUAGE TupleSections #-}
module Common where
import Prelude hiding (length, sequence, mapM)
import Control.Monad (join, msum)
import Data.Char
import Data.Maybe
import Data.Int
import Data.Vector hiding (map, (++))
import Control.Arrow
import Data.List (sort, groupBy)
import Data.Either
import Data.Function

import Data.Traversable as T

alpha :: String
alpha = " abcdefghijklmnopqrstuvwxyz0123456789.?,-:;'()";
-- alpha = map toLower " ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.?,-:;'()"

toChr :: Int8 -> Either String Int8
toChr x | x < 0 = Left $ "toChr: too small " ++ show x ++ " < 0"
        | fromIntegral x >= length toChr' = Left $ "toChr: too larg " ++ show x ++ " > " ++ show (length toChr')
        | otherwise = Right $ toChr' ! fromIntegral x

toChr' :: Vector Int8
toChr' = fromList . fmap (fromIntegral . ord) $ alpha

frChr :: Int8 -> Maybe Int8
frChr c = undefined -- join $ frChr' !? fromIntegral c

hush = either (const Nothing) Just

instance Traversable ((,) a) where
  sequence t = undefined

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