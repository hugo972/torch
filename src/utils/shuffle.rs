use rand::Rng;

pub struct ShuffleIter<I>
where
    I: Iterator,
{
    items: Option<Vec<I::Item>>,
    iter: I,
}

impl<I> Iterator for ShuffleIter<I>
where
    I: Iterator,
{
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        let items = match self.items {
            None => {
                self.items = Some(self.iter.by_ref().collect());
                self.items.as_mut().unwrap()
            }
            Some(ref mut items) => items,
        };

        if items.len() == 0 {
            None
        } else {
            Some(items.swap_remove(rand::rng().random_range(..items.len())))
        }
    }
}

pub trait ShuffleIterExt: Iterator + Sized {
    fn shuffle(self) -> ShuffleIter<Self> {
        ShuffleIter {
            items: None,
            iter: self,
        }
    }
}

impl<I: Iterator> ShuffleIterExt for I {}
