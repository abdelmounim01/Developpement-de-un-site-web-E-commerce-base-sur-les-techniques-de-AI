import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { DetailProductvisiteurComponent } from './detail-product.component';

describe('DetailProductComponent', () => {
  let component: DetailProductvisiteurComponent;
  let fixture: ComponentFixture<DetailProductvisiteurComponent>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ DetailProductvisiteurComponent ]
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(DetailProductvisiteurComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
