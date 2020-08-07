import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { ProductsComponentAdmin } from './products.component';

describe('ProductsComponent', () => {
  let component: ProductsComponentAdmin;
  let fixture: ComponentFixture<ProductsComponentAdmin>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ ProductsComponentAdmin ]
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(ProductsComponentAdmin);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
