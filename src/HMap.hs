{-# LANGUAGE FlexibleContexts #-}
-- module HMap where

import Prelude hiding (length, map, reverse)
import Control.Monad

import Data.Vector.Storable.MMap
-- import Data.Vector
import Foreign.Storable
import qualified Data.Vector.Storable as I
import Data.Vector
import System.Environment
import Data.Int
import Data.Ord
import Data.Char
import Control.Exception.Base
import qualified Data.List as DL

import Debug.Trace

import Common

g :: FilePath -> IO (Vector Int8)
g fp = fmap convert $ unsafeMMapVector fp Nothing

-- gc :: Vector Int8 -> Vector Char
-- gc = map (chr . fromIntegral)

f :: FilePath -> IO String
f fp = (fmap.fmap) (chr . fromIntegral) . fmap toList $ g fp


-- + n extra chars to the left and right.
-- n == 0 means: Just one char.
slicer :: Int -> Vector a -> Vector (Vector a)
slicer n v = generate (length v - n + 1) (\i -> slice i n v)

merge n = assert (odd n) $
          (m :) . join $ DL.zipWith (\x y -> [x,y]) [m-1, m - 2 .. 0] [m+1 .. n]
  where m = n `div` 2

unMergeIt l = assert (odd n) $
              fmap snd . DL.sortBy (comparing fst) $ DL.zip (merge n) l
  where n = DL.length l
        

unMergeIt (x:xs) = DL.reverse ls DL.++ x : rs
  where h (x:y:rest) = (x,y) : h rest
        h [] = []
        (ls, rs) = DL.unzip $ h xs

sorter :: Int -> Vector a -> Vector a
sorter n = assert (odd n) $
           \v -> assert (length v == n) $
                 map (v!) . fromList . merge $ n
  where m = n `div` 2
        score x = (abs (x - m), x)

unsorter :: Int -> Vector a -> Vector a
unsorter n = assert (odd n) $
             \v -> assert (length v == n)
                   fromList $ fmap (v !) (unMergeIt [0..n-1])

-- unsorter :: Int -> Vector a -> Vector a
-- unsorter n = assert (odd n) $
--             \v ->

rev = reverse . map reverse

context = context1 * 2 + 1
context1 = 2

search :: (?) -> Vector a -> Vector (Vector a) -> ?
search f needle haystack = scoreIt?

main = do
  (s:_) <- getArgs
--   return s
  print context
  print context1
  x <- liftM (force . map (force . sorter context) . slicer context . map frChr') (g s)
  print x